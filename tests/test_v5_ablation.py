from __future__ import annotations

import pytest

from qmt_quant.ablation import (
    leave_one_out_specs,
    pair_specs,
    single_factor_specs,
    summarize_ablation_metrics,
)
from qmt_quant.composites import CompositeSpec


def _spec() -> CompositeSpec:
    return CompositeSpec(
        name="demo",
        weights={"low_volatility": 0.4, "momentum": -0.3, "trend": -0.3},
    )


def test_ablation_specs_preserve_orientation_and_normalize():
    specs = leave_one_out_specs(_spec())
    without_low_vol = specs["low_volatility"]
    assert sum(abs(value) for value in without_low_vol.weights.values()) == pytest.approx(1.0)
    assert without_low_vol.weights["momentum"] < 0
    assert without_low_vol.weights["trend"] < 0


def test_single_and_pair_specs_are_deterministic():
    singles = single_factor_specs(_spec())
    assert singles["momentum"].weights == {"momentum": -1.0}
    pairs = pair_specs(_spec())
    assert set(pairs) == {
        ("low_volatility", "momentum"),
        ("low_volatility", "trend"),
        ("momentum", "trend"),
    }
    assert sum(abs(value) for value in pairs[("momentum", "trend")].weights.values()) == pytest.approx(1.0)


def test_ablation_summary_flags_negative_return_contribution():
    summary = summarize_ablation_metrics(
        {"total_return": -0.20, "sharpe": -0.1, "max_drawdown": -0.50},
        {
            "momentum": {"total_return": 0.05, "sharpe": 0.3, "max_drawdown": -0.30},
            "low_volatility": {"total_return": -0.30, "sharpe": -0.2, "max_drawdown": -0.55},
        },
    ).set_index("factor_removed")
    assert summary.loc["momentum", "return_contribution"] == pytest.approx(-0.25)
    assert summary.loc["low_volatility", "return_contribution"] == pytest.approx(0.10)
