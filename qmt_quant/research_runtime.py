from __future__ import annotations

from types import ModuleType

import pandas as pd

from .research_contracts import (
    assert_strict_research_metrics,
    coverage_or_fail,
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

    module._coverage_or_fail = _coverage_or_fail
    module._eligible_mask = _eligible_mask
    module._assert_strict_metrics = assert_strict_research_metrics
    module._stitch_fold_equity = stitch_fold_equity
    module._basic_alpha_gate = evaluate_basic_alpha_gate
