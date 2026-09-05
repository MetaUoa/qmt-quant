from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompositeSpec:
    name: str
    weights: dict[str, float]


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    clean = {str(k): float(v) for k, v in weights.items() if np.isfinite(v) and float(v) != 0.0}
    total = float(sum(abs(v) for v in clean.values()))
    if total <= 0:
        raise ValueError("composite requires at least one non-zero finite weight")
    return {k: v / total for k, v in clean.items()}


def equal_weight_spec(name: str, factors: list[str]) -> CompositeSpec:
    unique = list(dict.fromkeys(str(x) for x in factors))
    if not unique:
        raise ValueError("at least one factor is required")
    return CompositeSpec(name=name, weights=_normalize({factor: 1.0 for factor in unique}))


def ic_weight_spec(
    name: str,
    diagnostics: pd.DataFrame,
    *,
    factors: list[str] | None = None,
    metric: str = "mean_rank_ic",
    orientations: Mapping[str, int] | None = None,
    cap: float | None = None,
) -> CompositeSpec:
    if "factor" not in diagnostics or metric not in diagnostics:
        raise ValueError("diagnostics must contain factor and requested metric")
    frame = diagnostics.copy()
    if factors is not None:
        frame = frame.loc[frame["factor"].astype(str).isin(set(map(str, factors)))]
    weights: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        factor = str(getattr(row, "factor"))
        value = float(getattr(row, metric))
        if not np.isfinite(value):
            continue
        orientation = int((orientations or {}).get(factor, 1))
        if orientation == 0:
            continue
        value = abs(value) * orientation
        if cap is not None:
            value = float(np.clip(value, -abs(float(cap)), abs(float(cap))))
        weights[factor] = value
    return CompositeSpec(name=name, weights=_normalize(weights))


def default_v5_candidate_specs() -> list[CompositeSpec]:
    """Small, interpretable candidate set; intentionally not a parameter grid."""
    return [
        equal_weight_spec("defensive_quality", ["low_volatility", "liquidity_stability"]),
        equal_weight_spec(
            "defensive_reversal",
            ["low_volatility", "liquidity_stability", "short_reversal"],
        ),
        equal_weight_spec(
            "risk_controlled",
            [
                "low_volatility",
                "low_downside_risk",
                "liquidity_stability",
                "short_reversal",
            ],
        ),
    ]


def apply_composite(
    factor_panels: Mapping[str, pd.DataFrame],
    spec: CompositeSpec,
) -> pd.DataFrame:
    missing = sorted(set(spec.weights).difference(factor_panels))
    if missing:
        raise KeyError(f"missing composite factor panels: {', '.join(missing)}")
    weighted: pd.DataFrame | None = None
    available: pd.DataFrame | None = None
    for factor, weight in spec.weights.items():
        panel = factor_panels[factor]
        contribution = panel * float(weight)
        present = panel.notna().astype(float) * abs(float(weight))
        weighted = contribution if weighted is None else weighted.add(contribution, fill_value=0.0)
        available = present if available is None else available.add(present, fill_value=0.0)
    if weighted is None or available is None:
        raise ValueError("empty composite")
    return weighted.div(available.replace(0.0, np.nan)).where(available > 0.0)
