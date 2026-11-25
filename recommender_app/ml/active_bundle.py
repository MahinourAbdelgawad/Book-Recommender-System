"""Resolve the bundle used by the model service and activation tools."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "instance/ml/v001/bundle"
DEFAULT_POINTER = Path(os.environ.get(
    "MODEL_ACTIVE_BUNDLE_FILE", ROOT / "instance/ml/active_bundle.txt"
)).expanduser()


def active_bundle_path(pointer=DEFAULT_POINTER, fallback=DEFAULT_BUNDLE, environ=None):
    environ = os.environ if environ is None else environ
    override = environ.get("MODEL_BUNDLE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    pointer = Path(pointer)
    if pointer.exists():
        value = pointer.read_text().strip()
        if not value:
            raise ValueError(f"Active bundle pointer is empty: {pointer}")
        target = Path(value).expanduser()
        if not target.is_absolute():
            target = pointer.parent / target
        return target.resolve()
    return Path(fallback).expanduser().resolve()
