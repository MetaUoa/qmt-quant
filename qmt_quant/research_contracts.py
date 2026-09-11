from __future__ import annotations

from typing import Any, Mapping, cast

import pandas as pd

from .qmt_data import coverage_report
from .reference_data import ReferenceData


STRICT_MISSING_REFERENCE_KEYS = (
    "missing_limit_rows",
    "missing_st_dates",
    "missing_limit_dates",
    "missing_suspend_rows",
)


def coverage_or_fail(
    label: str,
    universe: list[str],
    bars: dict[str, pd.DataFrame],
    threshold: float,
) -> tuple[float, pd.DataFrame]:
    """Return symbol coverage and fail closed below the declared threshold."""
    report = coverage_report(universe, bars)
    ratio = float(report["loaded"].mean()) if not report.empty else 0.0
    if ratio < float(threshold):
        raise RuntimeError(
            f"{label} symbol coverage {ratio:.4%} is below required {threshold:.4%}"
        )
    return ratio, report


def research_signal_eligibility(
    *,
    raw_close: pd.DataFrame,
    amount: pd.DataFrame,
    dates: pd.DatetimeIndex,
    reference: ReferenceData,
    universe: list[str],
    min_price: float,
    min_amount: float,
    min_listing_sessions: int,
    amount_window: int,
    suspend: pd.DataFrame | None = None,
    require_same_day_tradable: bool = False,
    context: str = "research",
) -> pd.DataFrame:
    """Canonical research eligibility with an explicit tradability policy.

    Historical B/composite/factor diagnostics used raw-price + trailing-liquidity +
    PIT-membership/ST eligibility, while repaired V5-C additionally requires positive
    same-day turnover and an explicit ``suspendFlag == 0``. Keeping that distinction
    as an explicit argument removes duplicated implementations without silently
    rewriting already-frozen historical research semantics.
    """
    target_dates = pd.DatetimeIndex(dates).normalize().sort_values().unique()
    avg_amount = amount.rolling(amount_window, min_periods=amount_window).mean().reindex(
        target_dates
    )
    mask = raw_close.reindex(target_dates).ge(float(min_price)) & avg_amount.ge(
        float(min_amount)
    )
    if require_same_day_tradable:
        if suspend is None:
            raise ValueError("same-day tradability requires a suspension panel")
        same_day_amount = amount.reindex(target_dates).apply(pd.to_numeric, errors="coerce")
        same_day_suspend = suspend.reindex(target_dates).apply(
            pd.to_numeric, errors="coerce"
        )
        mask &= same_day_suspend.eq(0.0) & same_day_amount.gt(0.0)

    columns = mask.columns
    for ts in target_dates:
        if ts not in reference.st_dates:
            raise RuntimeError(f"missing ST snapshot on {context} date {ts.date()}")
        members = set(
            reference.filter_members(
                universe,
                ts,
                min_listing_sessions=min_listing_sessions,
            )
        )
        allowed = members.difference(reference.st_codes(ts))
        mask.loc[ts, :] &= columns.isin(allowed)
    return mask


def strict_signal_eligibility(
    *,
    raw_close: pd.DataFrame,
    amount: pd.DataFrame,
    suspend: pd.DataFrame,
    dates: pd.DatetimeIndex,
    reference: ReferenceData,
    universe: list[str],
    min_price: float,
    min_amount: float,
    min_listing_sessions: int,
    amount_window: int,
    context: str = "research",
) -> pd.DataFrame:
    """Canonical repaired V5-C same-day strict signal-date eligibility."""
    return research_signal_eligibility(
        raw_close=raw_close,
        amount=amount,
        suspend=suspend,
        dates=dates,
        reference=reference,
        universe=universe,
        min_price=min_price,
        min_amount=min_amount,
        min_listing_sessions=min_listing_sessions,
        amount_window=amount_window,
        require_same_day_tradable=True,
        context=context,
    )


def assert_strict_research_metrics(metrics: Mapping[str, object], label: str) -> None:
    """Refuse any research result with a missing strict execution reference."""
    for key in STRICT_MISSING_REFERENCE_KEYS:
        raw_value = cast(Any, metrics.get(key, 0))
        value = int(raw_value or 0)
        if value != 0:
            raise RuntimeError(f"{label} has {key}={value}; refusing research result")


def stitch_fold_equity(
    parts: list[pd.Series],
    *,
    initial_value: float | None = None,
) -> pd.Series:
    """Chain independent fold equity curves without dropping valid boundary sessions.

    ``initial_value`` should be the pre-trade NAV used to start each independent fold.
    Supplying it preserves each fold's first-session return. Without it, the historical
    first-row-normalization behavior is retained, but valid first dates are no longer
    unconditionally discarded.
    """
    stitched: list[pd.Series] = []
    chained = 1.0
    first_date: pd.Timestamp | None = None
    base = float(initial_value) if initial_value is not None else None
    if base is not None and base <= 0.0:
        raise ValueError("initial_value must be positive")

    for equity in parts:
        clean = equity.dropna().sort_index()
        if clean.empty:
            continue
        if first_date is None:
            first_date = pd.Timestamp(clean.index[0])
        denominator = base if base is not None else float(clean.iloc[0])
        if denominator <= 0.0:
            raise ValueError("fold equity baseline must be positive")
        normalized = clean / denominator * chained
        if normalized.empty:
            continue
        stitched.append(normalized)
        chained = float(normalized.iloc[-1])

    if not stitched:
        return pd.Series(dtype=float)

    out = pd.concat(stitched).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if base is not None and first_date is not None:
        baseline_index = first_date - pd.Timedelta(nanoseconds=1)
        baseline = pd.Series([1.0], index=pd.DatetimeIndex([baseline_index]))
        out = pd.concat([baseline, out]).sort_index()
    return out
