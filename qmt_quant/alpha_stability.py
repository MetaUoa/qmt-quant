from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StabilityScorePolicy:
    """Training-only stability diagnostics for factor weighting research."""

    min_years: int = 2
    min_dates_per_year: int = 6
    dispersion_floor: float = 1e-6


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
        icir = abs(mean_ic) / max(daily_std, float(cfg.dispersion_floor))
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
