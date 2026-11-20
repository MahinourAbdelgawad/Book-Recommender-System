"""Translate a local library into the model service's recommendation request."""

from dataclasses import dataclass
import math

import httpx


@dataclass(frozen=True)
class Prediction:
    book_id: str
    rating: float


class ModelServiceError(Exception):
    pass


def recommend(api_url, api_key, books, top_n=10, extra_excluded=()):
    ratings = {}
    rated_works = set()
    for book in books:
        identifier = book["goodreads_id"] or f"local:{book['id']}"
        work = book["work_id"] or f"entry:{identifier}"
        if book["rating"] is not None and work not in rated_works:
            ratings[identifier] = book["rating"]
            rated_works.add(work)
    excluded = set(extra_excluded) | {book["goodreads_id"] for book in books
                                     if book["goodreads_id"] and book["shelf"] != "to-read"}
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = httpx.post(f"{api_url.rstrip('/')}/v1/recommend", headers=headers,
                              json={"ratings": ratings, "exclude_book_ids": sorted(excluded), "top_n": top_n},
                              timeout=5)
        response.raise_for_status()
        payload = response.json()
        version = payload["model_version"]
        predictions = [Prediction(str(row["book_id"]), float(row["predicted_rating"]))
                       for row in payload["predictions"]]
        if (not isinstance(version, str) or not version or len(predictions) > top_n
                or any(not math.isfinite(row.rating) or not 1 <= row.rating <= 5 for row in predictions)):
            raise ValueError("Malformed recommendation response")
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise ModelServiceError("Recommendations are temporarily unavailable.") from error
    return version, predictions
