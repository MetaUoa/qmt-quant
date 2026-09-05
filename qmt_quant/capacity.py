from __future__ import annotations

import numpy as np
import pandas as pd


def exclude_bottom_liquidity(
    average_amount: pd.DataFrame,
    *,
    bottom_fraction: float,
) -> pd.DataFrame:
    """Return a date-local tradability mask excluding the least-liquid symbols.

    Ranking is cross-sectional on each date only. Missing liquidity stays False;
    nothing is forward/back filled across dates.
    """
    fraction = float(bottom_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("bottom_fraction must satisfy 0 <= x < 1")
    ranked = average_amount.rank(axis=1, method="average", pct=True, na_option="keep")
    mask = ranked.gt(fraction)
    return mask.where(average_amount.notna(), False).astype(bool)


def trade_capacity_report(
    trades: pd.DataFrame,
    daily_amount: pd.DataFrame,
    *,
    max_participation: float = 0.10,
) -> pd.DataFrame:
    required = {"date", "code", "notional"}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"trades missing columns: {', '.join(missing)}")
    threshold = float(max_participation)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("max_participation must satisfy 0 < x <= 1")

    rows = []
    for trade in trades.itertuples(index=False):
        ts = pd.Timestamp(getattr(trade, "date")).normalize()
        code = str(getattr(trade, "code"))
        notional = abs(float(getattr(trade, "notional")))
        amount = np.nan
        if ts in daily_amount.index and code in daily_amount.columns:
            amount = float(daily_amount.at[ts, code])
        valid = bool(np.isfinite(amount) and amount > 0.0)
        participation = notional / amount if valid else np.nan
        rows.append(
            {
                "date": ts,
                "code": code,
                "notional": notional,
                "daily_amount": amount,
                "participation": participation,
                "capacity_reference_present": valid,
                "capacity_pass": bool(valid and participation <= threshold),
            }
        )
    return pd.DataFrame(rows)


def summarize_capacity(report: pd.DataFrame, *, max_participation: float = 0.10) -> dict:
    if report is None or report.empty:
        return {
            "trade_rows": 0,
            "reference_coverage": 0.0,
            "capacity_pass_ratio": 0.0,
            "max_participation": None,
            "p95_participation": None,
            "threshold": float(max_participation),
            "passed": False,
        }
    present = report["capacity_reference_present"].astype(bool)
    participation = pd.to_numeric(report["participation"], errors="coerce")
    coverage = float(present.mean())
    pass_ratio = float(report["capacity_pass"].astype(bool).mean())
    valid = participation.dropna()
    max_value = float(valid.max()) if len(valid) else None
    p95 = float(valid.quantile(0.95)) if len(valid) else None
    passed = bool(coverage == 1.0 and pass_ratio == 1.0)
    return {
        "trade_rows": int(len(report)),
        "reference_coverage": coverage,
        "capacity_pass_ratio": pass_ratio,
        "max_participation": max_value,
        "p95_participation": p95,
        "threshold": float(max_participation),
        "passed": passed,
    }
