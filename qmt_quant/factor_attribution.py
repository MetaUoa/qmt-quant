from __future__ import annotations

from itertools import combinations
from typing import Mapping

import pandas as pd


def leave_one_out_attribution(
    full_return: float,
    without_factor_returns: Mapping[str, float],
) -> pd.DataFrame:
    """Measure each factor's marginal contribution from leave-one-out returns.

    Positive ``marginal_return`` means the full composite did better with the factor
    present. Negative values identify factors that dragged the realized portfolio.
    The function is deliberately agnostic to how returns were produced; callers are
    expected to use the same strict execution model for the full and ablated runs.
    """
    rows = []
    full = float(full_return)
    for factor, without_return in without_factor_returns.items():
        without = float(without_return)
        rows.append(
            {
                "factor": str(factor),
                "full_return": full,
                "without_factor_return": without,
                "marginal_return": full - without,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["factor", "full_return", "without_factor_return", "marginal_return"]
        )
    return pd.DataFrame(rows).sort_values(
        ["marginal_return", "factor"], ascending=[True, True]
    ).reset_index(drop=True)


def pair_interactions(
    single_factor_returns: Mapping[str, float],
    pair_returns: Mapping[tuple[str, str], float],
    *,
    baseline_return: float = 0.0,
) -> pd.DataFrame:
    """Estimate pair interaction beyond additive single-factor returns.

    Interaction = pair - single(A) - single(B) + baseline. Positive values indicate
    complementarity; negative values indicate destructive interaction.
    """
    rows = []
    baseline = float(baseline_return)
    singles = {str(name): float(value) for name, value in single_factor_returns.items()}
    for raw_pair, value in pair_returns.items():
        if len(raw_pair) != 2:
            raise ValueError("pair_returns keys must contain exactly two factor names")
        a, b = sorted((str(raw_pair[0]), str(raw_pair[1])))
        if a not in singles or b not in singles:
            raise KeyError(f"missing single-factor return for pair {a}, {b}")
        pair_value = float(value)
        rows.append(
            {
                "factor_a": a,
                "factor_b": b,
                "pair_return": pair_value,
                "factor_a_return": singles[a],
                "factor_b_return": singles[b],
                "baseline_return": baseline,
                "interaction_return": pair_value - singles[a] - singles[b] + baseline,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "factor_a",
                "factor_b",
                "pair_return",
                "factor_a_return",
                "factor_b_return",
                "baseline_return",
                "interaction_return",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["interaction_return", "factor_a", "factor_b"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def expected_pair_keys(factors: list[str] | tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    names = sorted({str(name) for name in factors})
    return tuple(combinations(names, 2))


def summarize_attribution(
    attribution: pd.DataFrame,
    interactions: pd.DataFrame | None = None,
) -> dict:
    """Return compact diagnostics highlighting the largest drag and interaction."""
    payload: dict[str, object] = {
        "factor_count": int(len(attribution)),
        "largest_drag_factor": None,
        "largest_drag_return": None,
        "largest_help_factor": None,
        "largest_help_return": None,
        "worst_interaction": None,
        "worst_interaction_return": None,
    }
    if not attribution.empty:
        ordered = attribution.sort_values("marginal_return")
        worst = ordered.iloc[0]
        best = ordered.iloc[-1]
        payload.update(
            {
                "largest_drag_factor": str(worst["factor"]),
                "largest_drag_return": float(worst["marginal_return"]),
                "largest_help_factor": str(best["factor"]),
                "largest_help_return": float(best["marginal_return"]),
            }
        )
    if interactions is not None and not interactions.empty:
        worst_pair = interactions.sort_values("interaction_return").iloc[0]
        payload["worst_interaction"] = [
            str(worst_pair["factor_a"]),
            str(worst_pair["factor_b"]),
        ]
        payload["worst_interaction_return"] = float(worst_pair["interaction_return"])
    return payload
