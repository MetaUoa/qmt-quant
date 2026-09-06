from __future__ import annotations

import math

import pandas as pd
import pytest

from qmt_quant.v5_gates import (
    FROZEN_V5_GATE_CONFIG,
    V5GateConfig,
    assert_v5_gate_config,
    evaluate_basic_alpha_gate,
)
from run_v5_c_nested_research import _basic_alpha_gate as historical_basic_alpha_gate


@pytest.mark.parametrize(
    ("metrics", "returns"),
    [
        ({"total_return": 0.10, "sharpe": 0.70, "max_drawdown": -0.30}, [0.2, 0.1, 0.0, -0.1, -0.3]),
        ({"total_return": -0.01, "sharpe": 0.70, "max_drawdown": -0.30}, [0.2, 0.1, 0.0, -0.1, -0.19]),
        ({"total_return": 0.10, "sharpe": 0.49, "max_drawdown": -0.30}, [0.2, 0.1, 0.0, -0.1, -0.19]),
        ({"total_return": 0.10, "sharpe": 0.50, "max_drawdown": -0.36}, [0.2, 0.1, 0.0, -0.1, -0.19]),
        ({"total_return": 0.10, "sharpe": 0.50, "max_drawdown": -0.35}, [0.2, -0.21, -0.22, -0.1, -0.19]),
    ],
)
def test_central_basic_gate_matches_historical_c1_contract(metrics, returns):
    folds = pd.DataFrame({"validation_return": returns})
    assert evaluate_basic_alpha_gate(metrics, folds) == historical_basic_alpha_gate(metrics, folds)


def test_default_policy_values_are_frozen_c1_values():
    policy = FROZEN_V5_GATE_CONFIG
    assert policy == V5GateConfig()
    assert policy.min_oos_total_return == 0.0
    assert policy.min_oos_sharpe == 0.50
    assert policy.max_oos_drawdown == 0.35
    assert policy.min_non_disastrous_folds == 4
    assert policy.disastrous_fold_loss == -0.20
    assert policy.min_stress_pass_ratio == 0.75
    assert policy.min_robustness_pass_ratio == 0.67


def test_stricter_gate_policy_is_allowed_without_mutation():
    policy = V5GateConfig(
        min_oos_total_return=0.01,
        min_oos_sharpe=0.60,
        max_oos_drawdown=0.30,
        min_non_disastrous_folds=5,
        disastrous_fold_loss=-0.10,
        min_stress_pass_ratio=0.80,
        min_robustness_pass_ratio=0.70,
    )
    assert assert_v5_gate_config(policy) is policy


@pytest.mark.parametrize(
    "policy",
    [
        V5GateConfig(min_oos_total_return=-0.01),
        V5GateConfig(min_oos_sharpe=0.49),
        V5GateConfig(max_oos_drawdown=0.36),
        V5GateConfig(min_non_disastrous_folds=3),
        V5GateConfig(disastrous_fold_loss=-0.21),
        V5GateConfig(min_stress_pass_ratio=0.74),
        V5GateConfig(min_robustness_pass_ratio=0.66),
        V5GateConfig(min_oos_sharpe=math.nan),
        V5GateConfig(max_oos_drawdown=math.inf),
    ],
)
def test_relaxed_or_non_finite_gate_policy_fails_closed(policy):
    with pytest.raises(RuntimeError):
        assert_v5_gate_config(policy)

    folds = pd.DataFrame({"validation_return": [0.1, 0.1, 0.1, 0.1]})
    metrics = {"total_return": 0.1, "sharpe": 1.0, "max_drawdown": -0.1}
    with pytest.raises(RuntimeError):
        evaluate_basic_alpha_gate(metrics, folds, config=policy)
