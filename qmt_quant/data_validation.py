from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def assess_free_data_manifest(
    manifest: dict,
    *,
    min_symbol_coverage: float = 0.98,
) -> dict:
    symbols = _safe_int(manifest.get("symbols"))
    adjusted = _safe_int(manifest.get("adjusted_symbols_cached"))
    raw = _safe_int(manifest.get("raw_symbols_cached"))
    adjusted_ratio = _ratio(adjusted, symbols)
    raw_ratio = _ratio(raw, symbols)
    errors = manifest.get("errors") or []
    failures: list[str] = []

    if str(manifest.get("source", "")).lower() != "baostock":
        failures.append("manifest source is not baostock")
    if symbols <= 0:
        failures.append("manifest contains no historical A-share symbols")
    if adjusted_ratio < min_symbol_coverage:
        failures.append(
            f"adjusted cache coverage {adjusted_ratio:.2%} < {min_symbol_coverage:.2%}"
        )
    if raw_ratio < min_symbol_coverage:
        failures.append(
            f"raw cache coverage {raw_ratio:.2%} < {min_symbol_coverage:.2%}"
        )

    error_kinds: dict[str, int] = {}
    for row in errors:
        if isinstance(row, dict):
            kind = str(row.get("kind", "unknown"))
            error_kinds[kind] = error_kinds.get(kind, 0) + 1

    return {
        "passed": not failures,
        "source": manifest.get("source"),
        "start": manifest.get("start"),
        "end": manifest.get("end"),
        "symbols": symbols,
        "adjusted_symbols_cached": adjusted,
        "raw_symbols_cached": raw,
        "adjusted_symbol_coverage_ratio": adjusted_ratio,
        "raw_symbol_coverage_ratio": raw_ratio,
        "strict_ready": bool(manifest.get("strict_ready", False)),
        "download_error_count": len(errors),
        "download_error_kinds": error_kinds,
        "thresholds": {"min_symbol_coverage": float(min_symbol_coverage)},
        "failures": failures,
    }


def assess_akshare_crosscheck(
    report: pd.DataFrame | None,
    *,
    min_pass_ratio: float = 0.80,
    min_compared: int = 5,
) -> dict:
    frame = pd.DataFrame() if report is None else report.copy()
    if frame.empty or "status" not in frame.columns:
        return {
            "ready": False,
            "rows": 0,
            "compared": 0,
            "passed": 0,
            "mismatched": 0,
            "unavailable": 0,
            "pass_ratio": 0.0,
            "thresholds": {
                "min_pass_ratio": float(min_pass_ratio),
                "min_compared": int(min_compared),
            },
        }

    status = frame["status"].astype(str)
    passed = int(status.eq("pass").sum())
    mismatched = int(status.eq("mismatch").sum())
    compared = passed + mismatched
    unavailable = int(len(frame) - compared)
    pass_ratio = _ratio(passed, compared)
    ready = bool(compared >= int(min_compared) and pass_ratio >= float(min_pass_ratio))
    return {
        "ready": ready,
        "rows": int(len(frame)),
        "compared": compared,
        "passed": passed,
        "mismatched": mismatched,
        "unavailable": unavailable,
        "pass_ratio": pass_ratio,
        "thresholds": {
            "min_pass_ratio": float(min_pass_ratio),
            "min_compared": int(min_compared),
        },
    }


def build_baseline_summary(
    metrics: dict,
    yearly_returns: pd.DataFrame | None = None,
) -> dict:
    yearly = pd.DataFrame() if yearly_returns is None else yearly_returns.copy()
    values = pd.Series(dtype=float)
    if not yearly.empty:
        if "return" in yearly.columns:
            values = pd.to_numeric(yearly["return"], errors="coerce").dropna()
        elif "yearly_return" in yearly.columns:
            values = pd.to_numeric(yearly["yearly_return"], errors="coerce").dropna()
        else:
            numeric = yearly.select_dtypes(include="number")
            if not numeric.empty:
                values = pd.to_numeric(numeric.iloc[:, -1], errors="coerce").dropna()

    summary = {
        "multiple": float(metrics.get("multiple", 0.0) or 0.0),
        "cagr": float(metrics.get("cagr", 0.0) or 0.0),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0) or 0.0),
        "sharpe": float(metrics.get("sharpe", 0.0) or 0.0),
        "calmar": float(metrics.get("calmar", 0.0) or 0.0),
        "trade_count": _safe_int(metrics.get("trade_count")),
        "symbol_coverage_ratio": float(metrics.get("symbol_coverage_ratio", 0.0) or 0.0),
        "positive_years": int((values > 0).sum()) if len(values) else None,
        "negative_years": int((values < 0).sum()) if len(values) else None,
        "best_year_return": float(values.max()) if len(values) else None,
        "worst_year_return": float(values.min()) if len(values) else None,
        "target_150x_reached": bool(float(metrics.get("multiple", 0.0) or 0.0) >= 150.0),
    }
    return summary


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
