from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .composites import CompositeSpec


def _renormalize(weights: Mapping[str, float]) -> dict[str, float]:
    cleaned = {str(name): float(value) for name, value in weights.items() if float(value) != 0.0}
    denominator = float(sum(abs(value) for value in cleaned.values()))
    if denominator <= 0.0:
        raise ValueError("ablation requires at least one non-zero remaining factor")
    return {name: value / denominator for name, value in cleaned.items()}


def leave_one_out_specs(spec: CompositeSpec) -> dict[str, CompositeSpec]:
    """Generate one frozen-weight ablation per factor.

    Remaining weights are rescaled by absolute weight so ranking magnitude stays
    comparable. No refitting is performed on OOS data.
    """
    weights = {str(name): float(value) for name, value in spec.weights.items()}
    if len(weights) < 2:
        raise ValueError("leave-one-out ablation requires at least two factors")
    result: dict[str, CompositeSpec] = {}
    for factor in sorted(weights):
        remaining = {name: value for name, value in weights.items() if name != factor}
        result[factor] = CompositeSpec(
            name=f"{spec.name}__without__{factor}",
            weights=_renormalize(remaining),
        )
    return result


def single_factor_specs(spec: CompositeSpec) -> dict[str, CompositeSpec]:
    result: dict[str, CompositeSpec] = {}
    for factor, weight in sorted(spec.weights.items()):
        orientation = 1.0 if float(weight) > 0.0 else -1.0
        result[str(factor)] = CompositeSpec(
            name=f"{spec.name}__single__{factor}",
            weights={str(factor): orientation},
        )
    return result


def pair_specs(spec: CompositeSpec) -> dict[tuple[str, str], CompositeSpec]:
    names = sorted(str(name) for name in spec.weights)
    result: dict[tuple[str, str], CompositeSpec] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            result[(first, second)] = CompositeSpec(
                name=f"{spec.name}__pair__{first}__{second}",
                weights=_renormalize(
                    {
                        first: float(spec.weights[first]),
                        second: float(spec.weights[second]),
                    }
                ),
            )
    return result


def summarize_ablation_metrics(
    full_metrics: Mapping[str, float],
    ablated_metrics: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Compare strict-backtest metrics after removing one factor at a time."""
    full_return = float(full_metrics.get("total_return", np.nan))
    full_sharpe = float(full_metrics.get("sharpe", np.nan))
    full_mdd = abs(float(full_metrics.get("max_drawdown", np.nan)))
    rows: list[dict] = []
    for factor, metrics in ablated_metrics.items():
        ablated_return = float(metrics.get("total_return", np.nan))
        ablated_sharpe = float(metrics.get("sharpe", np.nan))
        ablated_mdd = abs(float(metrics.get("max_drawdown", np.nan)))
        rows.append(
            {
                "factor_removed": str(factor),
                "full_return": full_return,
                "ablated_return": ablated_return,
                "return_contribution": full_return - ablated_return,
                "full_sharpe": full_sharpe,
                "ablated_sharpe": ablated_sharpe,
                "sharpe_contribution": full_sharpe - ablated_sharpe,
                "full_max_drawdown": full_mdd,
                "ablated_max_drawdown": ablated_mdd,
                "drawdown_cost": full_mdd - ablated_mdd,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["return_contribution", "sharpe_contribution", "factor_removed"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
