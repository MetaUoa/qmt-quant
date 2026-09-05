from __future__ import annotations

import pandas as pd

from qmt_quant.config import CostConfig, StrategyConfig
from qmt_quant.v5_stress_scenarios import build_v5_stress_scenarios, stress_pass_ratio


def test_stress_suite_is_small_predeclared_and_does_not_mutate_base():
    strategy = StrategyConfig(top_n=8, rebalance_days=5, execution_delay_sessions=1)
    costs = CostConfig()
    scenarios = build_v5_stress_scenarios(strategy, costs)
    names = [row.name for row in scenarios]
    assert names == [
        "base",
        "commission_1_5x",
        "commission_2x",
        "slippage_1_5x",
        "slippage_2x",
        "execution_delay_plus_1",
        "execution_delay_plus_2",
        "rebalance_10",
        "rebalance_20",
        "top_n_5",
        "top_n_12",
        "min_amount_30m",
        "min_amount_40m",
    ]
    assert strategy.top_n == 8
    assert strategy.rebalance_days == 5
    assert strategy.execution_delay_sessions == 1
    assert costs == CostConfig()


def test_stress_pass_ratio_excludes_reference_case():
    rows = pd.DataFrame(
        {
            "scenario": ["base", "a", "b", "c"],
            "total_return": [1.0, 0.1, -0.1, 0.2],
        }
    )
    assert abs(stress_pass_ratio(rows) - 2.0 / 3.0) < 1e-12
