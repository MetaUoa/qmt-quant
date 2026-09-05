from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _validate_fractions(fractions: Iterable[float]) -> tuple[float, ...]:
    values = tuple(sorted({float(value) for value in fractions}))
    if not values or any(value <= 0.0 or value >= 0.5 for value in values):
        raise ValueError("tail fractions must be between 0 and 0.5")
    return values


def tail_profile_for_date(
    factor: pd.Series,
    forward_return: pd.Series,
    *,
    fractions: Iterable[float] = (0.05, 0.10, 0.20),
    min_symbols: int = 50,
) -> dict[str, float | int]:
    """Evaluate extreme top/bottom tails for one cross-section.

    The function uses only the supplied factor values and ex-post forward-return labels.
    It is a diagnostic, never a portfolio-selection routine.
    """
    tails = _validate_fractions(fractions)
    pair = pd.concat(
        [
            pd.to_numeric(factor, errors="coerce").rename("factor"),
            pd.to_numeric(forward_return, errors="coerce").rename("forward"),
        ],
        axis=1,
    ).dropna()
    result: dict[str, float | int] = {"symbols": int(len(pair))}
    if len(pair) < int(min_symbols):
        for fraction in tails:
            key = int(round(fraction * 100))
            result[f"top_{key}_return"] = float("nan")
            result[f"bottom_{key}_return"] = float("nan")
            result[f"spread_{key}"] = float("nan")
        return result

    pct = pair["factor"].rank(method="average", pct=True)
    for fraction in tails:
        key = int(round(fraction * 100))
        top = pair.loc[pct > 1.0 - fraction, "forward"]
        bottom = pair.loc[pct <= fraction, "forward"]
        top_mean = float(top.mean()) if len(top) else float("nan")
        bottom_mean = float(bottom.mean()) if len(bottom) else float("nan")
        result[f"top_{key}_return"] = top_mean
        result[f"bottom_{key}_return"] = bottom_mean
        result[f"spread_{key}"] = (
            float(top_mean - bottom_mean)
            if np.isfinite(top_mean) and np.isfinite(bottom_mean)
            else float("nan")
        )
    return result


def tail_profile_observations(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    *,
    dates: Iterable[pd.Timestamp] | None = None,
    fractions: Iterable[float] = (0.05, 0.10, 0.20),
    min_symbols: int = 50,
) -> pd.DataFrame:
    common = factor.index.intersection(forward_return.index)
    if dates is not None:
        requested = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
        common = common.intersection(requested)
    rows: list[dict] = []
    for ts in common.sort_values():
        row = tail_profile_for_date(
            factor.loc[ts],
            forward_return.loc[ts],
            fractions=fractions,
            min_symbols=min_symbols,
        )
        if int(row["symbols"]) < int(min_symbols):
            continue
        rows.append({"date": pd.Timestamp(ts).normalize(), **row})
    return pd.DataFrame(rows)


def summarize_tail_profiles(
    observations: pd.DataFrame,
    *,
    fractions: Iterable[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    tails = _validate_fractions(fractions)
    rows: list[dict] = []
    for fraction in tails:
        key = int(round(fraction * 100))
        top = pd.to_numeric(observations.get(f"top_{key}_return"), errors="coerce")
        bottom = pd.to_numeric(observations.get(f"bottom_{key}_return"), errors="coerce")
        spread = pd.to_numeric(observations.get(f"spread_{key}"), errors="coerce")
        valid_spread = spread.dropna()
        rows.append(
            {
                "tail_fraction": fraction,
                "dates": int(len(valid_spread)),
                "mean_top_return": float(top.mean()) if top is not None else float("nan"),
                "mean_bottom_return": float(bottom.mean()) if bottom is not None else float("nan"),
                "mean_spread": float(valid_spread.mean()) if len(valid_spread) else float("nan"),
                "positive_spread_ratio": float((valid_spread > 0.0).mean())
                if len(valid_spread)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def tail_linearity_score(summary: pd.DataFrame) -> float:
    """Score whether stronger tail concentration produces larger spreads.

    A positive value means the absolute signal becomes stronger toward the extreme
    tail. This is descriptive only and must not be used on OOS rows to select a tail.
    """
    if summary.empty or len(summary) < 2:
        return float("nan")
    frame = summary.dropna(subset=["tail_fraction", "mean_spread"]).copy()
    if len(frame) < 2:
        return float("nan")
    x = -pd.to_numeric(frame["tail_fraction"], errors="coerce")
    y = pd.to_numeric(frame["mean_spread"], errors="coerce")
    value = x.corr(y)
    return float(value) if pd.notna(value) else float("nan")
