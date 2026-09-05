from __future__ import annotations

import numpy as np
import pandas as pd


def ic_correlation_matrix(observations: pd.DataFrame, *, horizon: int | None = None) -> pd.DataFrame:
    required = {"factor", "date", "rank_ic"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"factor observations missing columns: {', '.join(missing)}")
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if horizon is not None:
        if "horizon" not in frame.columns:
            raise ValueError("horizon column is required when horizon is specified")
        frame = frame.loc[frame["horizon"] == int(horizon)]
    pivot = frame.pivot_table(index="date", columns="factor", values="rank_ic", aggfunc="mean")
    return pivot.corr(method="spearman")


def panel_rank_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    min_symbols: int = 20,
) -> pd.Series:
    common_dates = left.index.intersection(right.index)
    rows = {}
    for ts in common_dates:
        pair = pd.concat([left.loc[ts], right.loc[ts]], axis=1, keys=["left", "right"]).dropna()
        if len(pair) < int(min_symbols):
            continue
        rows[pd.Timestamp(ts)] = float(pair["left"].corr(pair["right"], method="spearman"))
    return pd.Series(rows, dtype=float).sort_index()


def greedy_low_redundancy_selection(
    priority: pd.DataFrame,
    correlation: pd.DataFrame,
    *,
    score_column: str = "mean_rank_ic",
    max_abs_correlation: float = 0.80,
    max_factors: int = 5,
) -> pd.DataFrame:
    if "factor" not in priority or score_column not in priority:
        raise ValueError("priority must contain factor and score columns")
    ranked = priority.copy()
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="coerce")
    ranked = ranked.dropna(subset=[score_column]).sort_values(score_column, ascending=False)
    selected: list[str] = []
    rows: list[dict] = []
    for row in ranked.itertuples(index=False):
        factor = str(getattr(row, "factor"))
        if factor not in correlation.index:
            continue
        blockers = []
        for chosen in selected:
            value = correlation.loc[factor, chosen]
            if pd.notna(value) and abs(float(value)) > float(max_abs_correlation):
                blockers.append((chosen, float(value)))
        accepted = not blockers and len(selected) < int(max_factors)
        if accepted:
            selected.append(factor)
        rows.append(
            {
                "factor": factor,
                "score": float(getattr(row, score_column)),
                "accepted": bool(accepted),
                "max_abs_corr_to_selected": max((abs(v) for _, v in blockers), default=0.0),
                "blocked_by": ",".join(name for name, _ in blockers),
            }
        )
    return pd.DataFrame(rows)


def incremental_ic_gain(
    target_ic: pd.Series,
    candidate_ic: pd.Series,
    existing_ic: pd.DataFrame,
) -> dict:
    """Measure whether a candidate contributes information beyond an existing IC basket."""
    aligned = pd.concat(
        [
            pd.to_numeric(target_ic, errors="coerce").rename("target"),
            pd.to_numeric(candidate_ic, errors="coerce").rename("candidate"),
            existing_ic.apply(pd.to_numeric, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {"observations": 0, "candidate_target_corr": np.nan, "partial_corr": np.nan}
    candidate_target_corr = float(aligned["candidate"].corr(aligned["target"]))
    controls = [c for c in aligned.columns if c not in {"target", "candidate"}]
    if not controls:
        partial = candidate_target_corr
    else:
        x = np.column_stack([np.ones(len(aligned)), aligned[controls].to_numpy(dtype=float)])
        target_resid = aligned["target"].to_numpy(dtype=float) - x @ np.linalg.lstsq(
            x, aligned["target"].to_numpy(dtype=float), rcond=None
        )[0]
        candidate_resid = aligned["candidate"].to_numpy(dtype=float) - x @ np.linalg.lstsq(
            x, aligned["candidate"].to_numpy(dtype=float), rcond=None
        )[0]
        partial = float(pd.Series(target_resid).corr(pd.Series(candidate_resid)))
    return {
        "observations": int(len(aligned)),
        "candidate_target_corr": candidate_target_corr,
        "partial_corr": partial,
    }
