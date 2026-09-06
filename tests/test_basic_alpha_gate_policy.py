from __future__ import annotations

import pandas as pd
import pytest

from qmt_quant.v5_gates import V5GateConfig, evaluate_basic_alpha_gate
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
    policy = V5GateConfig()
    assert policy.min_oos_total_return == 0.0
    assert policy.min_oos_sharpe == 0.50
    assert policy.max_oos_drawdown == 0.35
    assert policy.min_non_disastrous_folds == 4
    assert policy.disastrous_fold_loss == -0.20
