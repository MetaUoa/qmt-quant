from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorOrientation:
    factor: str
    orientation: int
    mean_rank_ic: float
    worst_horizon_rank_ic: float
    mean_spread: float
    horizons: int
    dates: int


def _training_slice(observations: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    required = {"factor", "horizon", "date", "rank_ic", "top_bottom_spread"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"factor observations missing columns: {', '.join(missing)}")
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "factor", "horizon", "rank_ic"])
    if start is not None:
        frame = frame.loc[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(end)]
    return frame


def learn_factor_orientations(
    observations: pd.DataFrame,
    *,
    start=None,
    end=None,
    min_dates: int = 24,
    min_abs_rank_ic: float = 0.01,
    require_same_sign_across_horizons: bool = True,
) -> pd.DataFrame:
    """Learn factor direction from a training window only.

    Orientation is +1 for the raw factor, -1 for its inverse and 0 for a factor that
    does not clear the evidence threshold. No validation/test observations are used
    when ``end`` is supplied.
    """
    frame = _training_slice(observations, start, end)
    rows: list[dict] = []
    for factor, group in frame.groupby("factor", sort=True):
        per_horizon = group.groupby("horizon", as_index=False).agg(
            mean_rank_ic=("rank_ic", "mean"),
            dates=("date", "nunique"),
            mean_spread=("top_bottom_spread", "mean"),
        )
        if per_horizon.empty:
            continue
        total_dates = int(group["date"].nunique())
        means = pd.to_numeric(per_horizon["mean_rank_ic"], errors="coerce").dropna()
        mean_ic = float(means.mean()) if len(means) else 0.0
        worst_ic = float(means.min()) if len(means) else 0.0
        mean_spread = float(
            pd.to_numeric(group["top_bottom_spread"], errors="coerce").mean()
        )
        signs = {int(np.sign(x)) for x in means if abs(float(x)) >= min_abs_rank_ic}
        consistent = len(signs) <= 1
        orientation = 0
        if total_dates >= int(min_dates) and abs(mean_ic) >= float(min_abs_rank_ic):
            if not require_same_sign_across_horizons or consistent:
                orientation = 1 if mean_ic > 0 else -1
        rows.append(
            {
                "factor": str(factor),
                "orientation": int(orientation),
                "mean_rank_ic": mean_ic,
                "worst_horizon_rank_ic": worst_ic,
                "mean_spread": mean_spread,
                "horizons": int(per_horizon["horizon"].nunique()),
                "dates": total_dates,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["orientation", "mean_rank_ic"], ascending=[False, False]
    ).reset_index(drop=True)


def apply_orientation(frame: pd.DataFrame, orientation: int) -> pd.DataFrame:
    orientation = int(orientation)
    if orientation not in {-1, 0, 1}:
        raise ValueError("orientation must be -1, 0 or 1")
    if orientation == 0:
        return frame * np.nan
    return frame * float(orientation)


def duplicate_observation_groups(
    observations: pd.DataFrame,
    *,
    start=None,
    end=None,
    value_columns: Iterable[str] = ("rank_ic", "top_bottom_spread"),
    atol: float = 1e-12,
) -> list[list[str]]:
    """Detect duplicate factor research series using training observations only."""
    frame = _training_slice(observations, start, end)
    cols = [c for c in value_columns if c in frame.columns]
    if not cols:
        raise ValueError("no requested value columns are present")
    signatures: dict[str, np.ndarray] = {}
    keys = ["horizon", "date"]
    for factor, group in frame.groupby("factor", sort=True):
        ordered = group.sort_values(keys)
        signatures[str(factor)] = ordered[cols].to_numpy(dtype=float)
    factors = sorted(signatures)
    used: set[str] = set()
    groups: list[list[str]] = []
    for i, left in enumerate(factors):
        if left in used:
            continue
        peers = [left]
        for right in factors[i + 1 :]:
            if right in used:
                continue
            a, b = signatures[left], signatures[right]
            if a.shape == b.shape and np.allclose(a, b, equal_nan=True, atol=atol, rtol=0.0):
                peers.append(right)
                used.add(right)
        if len(peers) > 1:
            groups.append(peers)
            used.update(peers)
    return groups
