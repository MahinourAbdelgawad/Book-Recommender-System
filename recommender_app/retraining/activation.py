import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from recommender_app.ml.active_bundle import DEFAULT_POINTER, active_bundle_path
from recommender_app.ml.bundle import load_bundle
from .locking import exclusive_lock

class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ActivationError(RuntimeError):
    pass


def _atomic_pointer_write(pointer, contents):
    pointer = Path(pointer)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=pointer.parent,
            prefix=f".{pointer.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temp_name = temporary.name
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temp_name, pointer)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _api_json(api_url, path, api_key, *, method="GET", body=None, timeout=10):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        api_url.rstrip("/") + path,
        headers={
            **({"X-API-Key": api_key} if api_key else {}),
            **({"Content-Type": "application/json"} if data else {}),
        },
        data=data,
        method=method,
    )
    try:
        with build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ActivationError(f"Model service request failed at {path}: {error}") from error


def _restore_pointer(pointer, previous_contents):
    pointer = Path(pointer)
    if previous_contents is None:
        pointer.unlink(missing_ok=True)
    else:
        _atomic_pointer_write(pointer, previous_contents)


def _reload_and_verify(api_url, api_key, expected_version):
    reloaded = _api_json(api_url, "/internal/reload-model", api_key, method="POST")
    if reloaded.get("model_version") != expected_version:
        raise ActivationError("Model service reloaded an unexpected bundle version")
    health = _api_json(api_url, "/health", api_key)
    if health.get("status") != "ok" or health.get("model_version") != expected_version:
        raise ActivationError("Model service health check did not confirm the selected bundle")
    smoke = _api_json(
        api_url, "/v1/recommend", api_key, method="POST",
        body={"ratings": {}, "exclude_book_ids": [], "top_n": 1},
    )
    predictions = smoke.get("predictions")
    if smoke.get("model_version") != expected_version or not isinstance(predictions, list) \
            or len(predictions) != 1:
        raise ActivationError("Model service prediction smoke check failed")
    try:
        rating = float(predictions[0]["predicted_rating"])
    except (KeyError, TypeError, ValueError) as error:
        raise ActivationError("Model service returned an invalid smoke prediction") from error
    if not math.isfinite(rating) or not 1 <= rating <= 5:
        raise ActivationError("Model service returned an out-of-range smoke prediction")
    return health


def check_active_service(api_url, api_key, expected_version):
    parsed = urlsplit(api_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost", "127.0.0.1", "::1"
    } or parsed.username or parsed.password or parsed.path not in {"", "/"} \
            or parsed.query or parsed.fragment:
        raise ActivationError("Model API URL must point to a loopback host")
    health = _api_json(api_url, "/health", api_key)
    if health.get("status") != "ok" or health.get("model_version") != expected_version:
        raise ActivationError("Model service is not serving the selected active bundle")
    return health


def activate_candidate(candidate_bundle, output_root, api_url, api_key, pointer=DEFAULT_POINTER):
    """Switch the active pointer; restore and reload the previous bundle on any failure."""
    if os.environ.get("MODEL_BUNDLE_DIR"):
        raise ActivationError(
            "Unset MODEL_BUNDLE_DIR before activation; it overrides the managed active pointer"
        )
    candidate = Path(candidate_bundle).resolve()
    root = Path(output_root).resolve()
    if candidate == root or root not in candidate.parents:
        raise ActivationError("Candidate bundle must be inside the configured candidate output folder")

    try:
        _, manifest = load_bundle(candidate)
    except Exception as error:
        raise ActivationError(f"Candidate bundle failed validation: {error}") from error
    if manifest.get("selection", {}).get("passed") is not True:
        raise ActivationError("Candidate did not pass its validation gate")
    if not manifest.get("retraining", {}).get("included_ratings"):
        raise ActivationError("Candidate bundle is missing its included-rating record")

    candidate_version = manifest["version"]
    if manifest.get("selection", {}).get("candidate_version") != candidate_version:
        raise ActivationError("Candidate report version does not match the bundle version")
    pointer = Path(pointer)
    with exclusive_lock(pointer.with_suffix(".activation.lock")):
        active = active_bundle_path(pointer=pointer)
        if active == candidate:
            raise ActivationError("Candidate bundle is already active")
        active_model, active_manifest = load_bundle(active)
        del active_model
        previous_version = active_manifest["version"]
        check_active_service(api_url, api_key, previous_version)
        selection = manifest.get("selection", {})
        if (selection.get("active_version") != previous_version
                or selection.get("active_weights_sha256") != active_manifest["weights_sha256"]):
            raise ActivationError(
                "Candidate was trained against a different active bundle; retrain it before activation"
            )
        if manifest.get("dataset_sha256") != active_manifest.get("dataset_sha256"):
            if not (previous_version == "v001"
                    and selection.get("validation_protocol") == 1
                    and selection.get("dataset_sha256") == manifest["dataset_sha256"]
                    and selection.get("active_dataset_sha256") == active_manifest["dataset_sha256"]):
                raise ActivationError("Candidate has no verified transition from the active dataset")

        previous_contents = pointer.read_text() if pointer.exists() else None
        _atomic_pointer_write(pointer, str(candidate))
        try:
            health = _reload_and_verify(api_url, api_key, candidate_version)
        except Exception as activation_error:
            try:
                _restore_pointer(pointer, previous_contents)
            except Exception as pointer_error:
                raise ActivationError(
                    "Candidate activation failed and the active pointer could not be restored. "
                    f"Restore it to {active} and restart the model service. "
                    f"Activation error: {activation_error}; pointer error: {pointer_error}"
                ) from pointer_error
            try:
                _reload_and_verify(api_url, api_key, previous_version)
            except Exception as rollback_error:
                raise ActivationError(
                    "Candidate activation failed and the model service could not confirm rollback. "
                    f"The active pointer was restored to {active}; restart the model service and "
                    f"check /health. Activation error: {activation_error}; rollback error: {rollback_error}"
                ) from rollback_error
            raise ActivationError(
                f"Candidate activation failed; restored {previous_version}: {activation_error}"
            ) from activation_error

    return {
        "activated": True,
        "previous_version": previous_version,
        "active_version": candidate_version,
        "active_bundle": str(candidate),
        "health": health,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "instance/ml/candidates")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    api_key = os.environ.get("MODEL_API_KEY", "")
    try:
        result = activate_candidate(
            args.candidate_bundle, args.output_root, args.api_url, api_key
        )
    except (ActivationError, OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
