from __future__ import annotations

from types import ModuleType

import pandas as pd

from .research_contracts import (
    assert_strict_research_metrics,
    coverage_or_fail,
    research_signal_eligibility,
    stitch_fold_equity,
    strict_signal_eligibility,
)
from .v5_gates import evaluate_basic_alpha_gate


def install_v5_c_contracts(module: ModuleType) -> None:
    """Bind the legacy C1 runner surface to canonical strict research semantics.

    The nested selection/factor implementation remains in the existing runner. Only
    shared data-quality, eligibility, strict-reference, fold stitching and Basic Gate
    helpers are replaced. Keeping the adapter explicit makes the migration auditable
    without duplicating the full nested algorithm in another script.
    """

    def _coverage_or_fail(label, universe, bars, threshold) -> float:
        ratio, _ = coverage_or_fail(label, universe, bars, threshold)
        return ratio

    def _eligible_mask(
        *,
        raw_close: pd.DataFrame,
        amount: pd.DataFrame,
        suspend: pd.DataFrame,
        dates: pd.DatetimeIndex,
        reference,
        universe: list[str],
        min_price: float,
        min_amount: float,
        min_listing_sessions: int,
        amount_window: int,
    ) -> pd.DataFrame:
        return strict_signal_eligibility(
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
            context="C research",
        )

    setattr(module, "_coverage_or_fail", _coverage_or_fail)
    setattr(module, "_eligible_mask", _eligible_mask)
    setattr(module, "_assert_strict_metrics", assert_strict_research_metrics)
    setattr(module, "_stitch_fold_equity", stitch_fold_equity)
    setattr(module, "_basic_alpha_gate", evaluate_basic_alpha_gate)


def install_legacy_v5_research_contracts(
    module: ModuleType,
    *,
    context: str,
) -> None:
    """Centralize B/composite runner helpers without rewriting frozen semantics.

    Those historical runners intentionally used trailing-liquidity + raw-price +
    PIT-membership/ST eligibility, rather than repaired C1 same-day tradability.
    The canonical helper makes that policy explicit so the duplicate implementations
    can be removed one runner at a time without silently changing old research.
    """

    def _eligible_mask(
        *,
        raw_close: pd.DataFrame,
        amount: pd.DataFrame,
        dates: pd.DatetimeIndex,
        reference,
        universe: list[str],
        min_price: float,
        min_amount: float,
        min_listing_sessions: int,
        amount_window: int,
    ) -> pd.DataFrame:
        return research_signal_eligibility(
            raw_close=raw_close,
            amount=amount,
            dates=dates,
            reference=reference,
            universe=universe,
            min_price=min_price,
            min_amount=min_amount,
            min_listing_sessions=min_listing_sessions,
            amount_window=amount_window,
            require_same_day_tradable=False,
            context=context,
        )

    if hasattr(module, "_coverage_or_fail"):
        setattr(module, "_coverage_or_fail", coverage_or_fail)
    if hasattr(module, "_eligible_mask"):
        setattr(module, "_eligible_mask", _eligible_mask)
    if hasattr(module, "_assert_strict_metrics"):
        setattr(module, "_assert_strict_metrics", assert_strict_research_metrics)
    if hasattr(module, "_stitch_fold_equity"):
        setattr(module, "_stitch_fold_equity", stitch_fold_equity)
