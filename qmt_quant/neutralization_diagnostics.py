from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"variant", "factor", "date", "rank_ic"}


def summarize_neutralization_variants(
    observations: pd.DataFrame,
    *,
    start,
    end,
) -> pd.DataFrame:
    """Summarize training/validation evidence without selecting a winner.

    Selection is intentionally left to the nested inner-validation layer. This helper
    only produces comparable diagnostics for raw/liquidity/industry/combined variants.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(observations.columns))
    if missing:
        raise ValueError(f"neutralization observations missing columns: {', '.join(missing)}")
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
    frame = frame.dropna(subset=["variant", "factor", "date", "rank_ic"])
    frame = frame.loc[
        frame["date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
    ]
    rows: list[dict] = []
    for (variant, factor), group in frame.groupby(["variant", "factor"], sort=True):
        values = group["rank_ic"].astype(float)
        mean_ic = float(values.mean())
        std_ic = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        icir = mean_ic / std_ic if std_ic > 0.0 else (np.inf if mean_ic > 0 else -np.inf if mean_ic < 0 else 0.0)
        rows.append(
            {
                "variant": str(variant),
                "factor": str(factor),
                "mean_rank_ic": mean_ic,
                "abs_mean_rank_ic": abs(mean_ic),
                "rank_ic_std": std_ic,
                "icir": float(icir),
                "positive_ic_fraction": float((values > 0.0).mean()),
                "dates": int(group["date"].nunique()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "variant",
                "factor",
                "mean_rank_ic",
                "abs_mean_rank_ic",
                "rank_ic_std",
                "icir",
                "positive_ic_fraction",
                "dates",
            ]
        )
    return pd.DataFrame(rows).sort_values(["variant", "factor"]).reset_index(drop=True)


def aggregate_variant_quality(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate factor diagnostics per variant without using holdout performance."""
    required = {"variant", "factor", "abs_mean_rank_ic", "positive_ic_fraction", "dates"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"neutralization summary missing columns: {', '.join(missing)}")
    if summary.empty:
        return pd.DataFrame(columns=["variant", "factors", "mean_abs_rank_ic", "mean_positive_ic_fraction", "min_dates"])
    return (
        summary.groupby("variant", as_index=False)
        .agg(
            factors=("factor", "nunique"),
            mean_abs_rank_ic=("abs_mean_rank_ic", "mean"),
            mean_positive_ic_fraction=("positive_ic_fraction", "mean"),
            min_dates=("dates", "min"),
        )
        .sort_values(["mean_abs_rank_ic", "variant"], ascending=[False, True])
        .reset_index(drop=True)
    )
