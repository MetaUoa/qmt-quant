from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .composites import CompositeSpec, ic_weight_spec
from .factor_orthogonality import greedy_low_redundancy_selection, ic_correlation_matrix
from .factor_selection import duplicate_observation_groups, learn_factor_orientations


DEFAULT_SAFE_FACTORS = (
    "low_volatility",
    "low_downside_risk",
    "liquidity_stability",
    "short_reversal",
    "momentum_20_5",
    "momentum_60_5",
    "momentum_120_5",
    "trend_quality",
    "trend_persistence",
)


@dataclass(frozen=True)
class TrainingCompositeSelection:
    train_start: str
    train_end: str
    spec: CompositeSpec
    selected_factors: tuple[str, ...]
    orientations: dict[str, int]
    duplicate_groups: tuple[tuple[str, ...], ...]
    correlation_horizon: int

    def to_dict(self) -> dict:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "composite": self.spec.name,
            "weights": dict(self.spec.weights),
            "selected_factors": list(self.selected_factors),
            "orientations": dict(self.orientations),
            "duplicate_groups": [list(group) for group in self.duplicate_groups],
            "correlation_horizon": int(self.correlation_horizon),
        }


def select_training_composite(
    observations: pd.DataFrame,
    *,
    train_start,
    train_end,
    allowed_factors: tuple[str, ...] = DEFAULT_SAFE_FACTORS,
    correlation_horizon: int = 20,
    min_abs_rank_ic: float = 0.01,
    max_abs_correlation: float = 0.80,
    min_factors: int = 2,
    max_factors: int = 4,
    weight_metric_cap: float = 0.10,
) -> TrainingCompositeSelection:
    """Freeze one V5 composite using training observations only.

    Raw liquidity is intentionally excluded from ``DEFAULT_SAFE_FACTORS`` until
    capacity/liquidity stress testing demonstrates that the apparent low-liquidity
    premium is executable. Validation rows never participate in direction, duplicate,
    redundancy or weight decisions.
    """
    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    if end < start:
        raise ValueError("train_end must not be before train_start")

    learned = learn_factor_orientations(
        observations,
        start=start,
        end=end,
        min_abs_rank_ic=min_abs_rank_ic,
    )
    allowed = set(allowed_factors)
    learned = learned.loc[
        learned["factor"].isin(allowed) & learned["orientation"].ne(0)
    ].copy()
    if learned.empty:
        raise RuntimeError("no safe factor cleared the training orientation gate")

    duplicates = duplicate_observation_groups(observations, start=start, end=end)
    duplicate_drop: set[str] = set()
    for group in duplicates:
        eligible = sorted(set(group).intersection(allowed))
        if len(eligible) > 1:
            duplicate_drop.update(eligible[1:])
    if duplicate_drop:
        learned = learned.loc[~learned["factor"].isin(duplicate_drop)].copy()

    learned["selection_score"] = learned["mean_rank_ic"].abs()
    correlation = ic_correlation_matrix(
        observations,
        horizon=int(correlation_horizon),
        start=start,
        end=end,
    )
    redundancy = greedy_low_redundancy_selection(
        learned[["factor", "selection_score"]],
        correlation,
        score_column="selection_score",
        max_abs_correlation=max_abs_correlation,
        max_factors=max_factors,
    )
    selected = redundancy.loc[redundancy["accepted"], "factor"].astype(str).tolist()
    if len(selected) < int(min_factors):
        raise RuntimeError(
            f"only {len(selected)} safe low-redundancy factors survived; require {min_factors}"
        )

    oriented = {
        str(row.factor): int(row.orientation)
        for row in learned.itertuples(index=False)
        if str(row.factor) in selected
    }
    diagnostics = learned.loc[learned["factor"].isin(selected), ["factor", "mean_rank_ic"]]
    spec = ic_weight_spec(
        "training_ic_low_redundancy",
        diagnostics,
        factors=selected,
        metric="mean_rank_ic",
        orientations=oriented,
        cap=weight_metric_cap,
    )
    return TrainingCompositeSelection(
        train_start=str(start.date()),
        train_end=str(end.date()),
        spec=spec,
        selected_factors=tuple(selected),
        orientations=oriented,
        duplicate_groups=tuple(tuple(group) for group in duplicates),
        correlation_horizon=int(correlation_horizon),
    )
