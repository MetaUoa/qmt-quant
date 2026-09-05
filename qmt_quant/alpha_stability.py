from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .composites import CompositeSpec
from .v5_selector import TrainingCompositeSelection


@dataclass(frozen=True)
class StabilityScorePolicy:
    """Training-only stability diagnostics for factor weighting research."""

    min_years: int = 2
    min_dates_per_year: int = 6
    dispersion_floor: float = 1e-6
    icir_cap: float = 5.0


def factor_stability_scores(
    observations: pd.DataFrame,
    *,
    start,
    end,
    allowed_factors: tuple[str, ...],
    policy: StabilityScorePolicy | None = None,
) -> pd.DataFrame:
    """Score factor IC strength and temporal stability using training rows only.

    This function is intentionally research-only: it does not alter the current C1
    selector. Validation/holdout observations are excluded by the explicit end date.
    """
    cfg = policy or StabilityScorePolicy()
    if float(cfg.icir_cap) <= 0.0:
        raise ValueError("icir_cap must be positive")
    required = {"factor", "date", "rank_ic"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"factor observations missing columns: {', '.join(missing)}")
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
    frame = frame.dropna(subset=["date", "factor", "rank_ic"])
    frame = frame.loc[
        frame["date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        & frame["factor"].isin(set(allowed_factors))
    ]
    rows: list[dict] = []
    for factor, group in frame.groupby("factor", sort=True):
        group = group.copy()
        group["year"] = group["date"].dt.year
        per_year = group.groupby("year")["rank_ic"].agg(["mean", "count"])
        per_year = per_year.loc[per_year["count"] >= int(cfg.min_dates_per_year)]
        if len(per_year) < int(cfg.min_years):
            continue
        mean_ic = float(group["rank_ic"].mean())
        orientation = 1 if mean_ic >= 0 else -1
        aligned_year_ic = per_year["mean"].astype(float) * float(orientation)
        daily_std = float(group["rank_ic"].std(ddof=1))
        if not np.isfinite(daily_std):
            daily_std = 0.0
        raw_icir = abs(mean_ic) / max(daily_std, float(cfg.dispersion_floor))
        icir = min(float(raw_icir), float(cfg.icir_cap))
        positive_year_fraction = float((aligned_year_ic > 0.0).mean())
        worst_aligned_year_ic = float(aligned_year_ic.min())
        stability_score = float(abs(mean_ic) * (1.0 + icir) * positive_year_fraction)
        rows.append(
            {
                "factor": str(factor),
                "orientation": int(orientation),
                "mean_rank_ic": mean_ic,
                "icir": float(icir),
                "positive_year_fraction": positive_year_fraction,
                "worst_aligned_year_ic": worst_aligned_year_ic,
                "years": int(len(per_year)),
                "dates": int(group["date"].nunique()),
                "stability_score": stability_score,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "factor",
                "orientation",
                "mean_rank_ic",
                "icir",
                "positive_year_fraction",
                "worst_aligned_year_ic",
                "years",
                "dates",
                "stability_score",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["stability_score", "factor"], ascending=[False, True]
    ).reset_index(drop=True)


def stability_reweight_selection(
    observations: pd.DataFrame,
    selection: TrainingCompositeSelection,
    *,
    start,
    end,
    policy: StabilityScorePolicy | None = None,
) -> TrainingCompositeSelection:
    """Reweight an already-selected C1 factor set using training-only stability.

    Factor inclusion, redundancy pruning and orientation remain exactly those of the
    supplied C1 selection. Only the magnitudes are replaced by bounded stability
    scores estimated on the same explicit training window. Missing stability evidence
    fails closed rather than silently falling back to mean-IC weights.
    """
    selected = tuple(selection.selected_factors)
    if not selected:
        raise RuntimeError("cannot stability-reweight an empty C1 selection")
    scores = factor_stability_scores(
        observations,
        start=start,
        end=end,
        allowed_factors=selected,
        policy=policy,
    )
    by_factor = scores.set_index("factor") if not scores.empty else pd.DataFrame()
    missing = [factor for factor in selected if factor not in by_factor.index]
    if missing:
        raise RuntimeError(
            "missing training stability evidence for selected factors: " + ", ".join(missing)
        )

    raw_weights: dict[str, float] = {}
    for factor in selected:
        orientation = int(selection.orientations.get(factor, 0))
        if orientation not in {-1, 1}:
            raise RuntimeError(f"selected factor {factor} has invalid orientation {orientation}")
        score = float(by_factor.loc[factor, "stability_score"])
        if not np.isfinite(score) or score <= 0.0:
            raise RuntimeError(f"selected factor {factor} has non-positive stability score")
        raw_weights[factor] = float(orientation) * score

    total = float(sum(abs(value) for value in raw_weights.values()))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("stability weighting produced no finite positive magnitude")
    weights = {factor: value / total for factor, value in raw_weights.items()}
    return TrainingCompositeSelection(
        train_start=str(pd.Timestamp(start).date()),
        train_end=str(pd.Timestamp(end).date()),
        spec=CompositeSpec(name="training_ic_stability_weighted", weights=weights),
        selected_factors=selection.selected_factors,
        orientations=dict(selection.orientations),
        duplicate_groups=selection.duplicate_groups,
        correlation_horizon=int(selection.correlation_horizon),
    )
