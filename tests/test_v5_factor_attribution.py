from __future__ import annotations

import pytest

from qmt_quant.factor_attribution import (
    expected_pair_keys,
    leave_one_out_attribution,
    pair_interactions,
    summarize_attribution,
)


def test_leave_one_out_identifies_drag_and_help():
    result = leave_one_out_attribution(
        -0.10,
        {
            "low_volatility": -0.20,
            "inverse_momentum": 0.05,
        },
    )
    rows = result.set_index("factor")
    assert rows.loc["low_volatility", "marginal_return"] == pytest.approx(0.10)
    assert rows.loc["inverse_momentum", "marginal_return"] == pytest.approx(-0.15)
    summary = summarize_attribution(result)
    assert summary["largest_drag_factor"] == "inverse_momentum"
    assert summary["largest_help_factor"] == "low_volatility"


def test_pair_interaction_is_incremental_to_singles():
    result = pair_interactions(
        {"a": 0.04, "b": 0.03, "c": -0.01},
        {("a", "b"): 0.10, ("a", "c"): 0.00},
    )
    rows = result.set_index(["factor_a", "factor_b"])
    assert rows.loc[("a", "b"), "interaction_return"] == pytest.approx(0.03)
    assert rows.loc[("a", "c"), "interaction_return"] == pytest.approx(-0.03)


def test_expected_pair_keys_are_deterministic():
    assert expected_pair_keys(["b", "a", "c", "a"]) == (
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    )
