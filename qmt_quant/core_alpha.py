from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .v5_selector import TrainingCompositeSelection, select_training_composite


CORE_ALPHA_FACTORS = (
    "liquidity_stability",
    "low_volatility",
    "low_downside_risk",
    "short_reversal",
)

CHALLENGER_FACTORS = (
    "momentum_120_5",
    "trend_quality",
    "residual_relative_strength_60_5",
)

# B1-B6 established that 20-session momentum was the largest mean drag.
# Keep it explicit so later code cannot silently reintroduce it into the core pool.
EXCLUDED_CORE_FACTORS = ("momentum_20_5", "momentum_60_5", "relative_strength_60_5")


@dataclass(frozen=True)
class CoreAlphaPolicy:
    min_abs_rank_ic: float = 0.01
    max_abs_correlation: float = 0.80
    min_factors: int = 2
    max_factors: int = 4
    weight_metric_cap: float = 0.10
    include_challengers: bool = False
    stability_weighting: bool = False

    @property
    def allowed_factors(self) -> tuple[str, ...]:
        return CORE_ALPHA_FACTORS + (CHALLENGER_FACTORS if self.include_challengers else ())


def select_core_alpha(
    observations: pd.DataFrame,
    *,
    train_start,
    train_end,
    policy: CoreAlphaPolicy | None = None,
    correlation_horizon: int = 20,
) -> TrainingCompositeSelection:
    """Freeze the post-B-research alpha stack using training evidence only.

    The core pool intentionally excludes the momentum factors identified as unstable
    in B1-B6. Directions, redundancy and weights are learned only from the supplied
    training window; no validation observations are consulted here. C7 can replace
    only the selected factors' weight magnitudes with bounded training-only stability
    scores while preserving C1 inclusion, orientation and redundancy decisions.
    """
    cfg = policy or CoreAlphaPolicy()
    selection = select_training_composite(
        observations,
        train_start=train_start,
        train_end=train_end,
        allowed_factors=cfg.allowed_factors,
        correlation_horizon=correlation_horizon,
        min_abs_rank_ic=cfg.min_abs_rank_ic,
        max_abs_correlation=cfg.max_abs_correlation,
        min_factors=cfg.min_factors,
        max_factors=cfg.max_factors,
        weight_metric_cap=cfg.weight_metric_cap,
    )
    forbidden = set(EXCLUDED_CORE_FACTORS).intersection(selection.selected_factors)
    if forbidden:
        raise RuntimeError(f"excluded core factors were selected: {sorted(forbidden)}")
    if cfg.stability_weighting:
        from .alpha_stability import stability_reweight_selection

        selection = stability_reweight_selection(
            observations,
            selection,
            start=train_start,
            end=train_end,
        )
    return selection


def assert_core_selection(selection: TrainingCompositeSelection) -> None:
    allowed = set(CORE_ALPHA_FACTORS) | set(CHALLENGER_FACTORS)
    selected = set(selection.selected_factors)
    unknown = selected.difference(allowed)
    if unknown:
        raise RuntimeError(f"selection contains factors outside the C1 policy: {sorted(unknown)}")
    if selected.intersection(EXCLUDED_CORE_FACTORS):
        raise RuntimeError("selection reintroduced an explicitly excluded momentum factor")
