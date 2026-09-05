from __future__ import annotations

import json

import numpy as np

import run_v5_composite_oos


_ORIGINAL_JSON_DUMP = json.dump


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _safe_json_dump(obj, fp, *args, **kwargs):
    kwargs.setdefault("default", _json_default)
    return _ORIGINAL_JSON_DUMP(obj, fp, *args, **kwargs)


def main() -> int:
    original = run_v5_composite_oos.json.dump
    run_v5_composite_oos.json.dump = _safe_json_dump
    try:
        return run_v5_composite_oos.main()
    finally:
        run_v5_composite_oos.json.dump = original


if __name__ == "__main__":
    raise SystemExit(main())
