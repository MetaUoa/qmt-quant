from __future__ import annotations

from dataclasses import dataclass, replace

from .config import CostConfig, StrategyConfig


@dataclass(frozen=True)
class V5StressScenario:
    name: str
    strategy: StrategyConfig
    costs: CostConfig
    category: str


def build_v5_stress_scenarios(
    strategy: StrategyConfig,
    costs: CostConfig,
) -> list[V5StressScenario]:
    """Return a small, pre-declared robustness suite without parameter search.

    These scenarios are perturbations of an already frozen V5 strategy. They must
    never be used to select the OOS strategy; they only evaluate fragility after the
    stock-selection alpha gate has been measured.
    """
    scenarios = [
        V5StressScenario("base", strategy, costs, "reference"),
        V5StressScenario(
            "commission_1_5x",
            strategy,
            replace(
                costs,
                commission_rate=costs.commission_rate * 1.5,
                min_commission=costs.min_commission * 1.5,
            ),
            "cost",
        ),
        V5StressScenario(
            "commission_2x",
            strategy,
            replace(
                costs,
                commission_rate=costs.commission_rate * 2.0,
                min_commission=costs.min_commission * 2.0,
            ),
            "cost",
        ),
        V5StressScenario(
            "slippage_1_5x",
            strategy,
            replace(costs, slippage_bps=costs.slippage_bps * 1.5),
            "cost",
        ),
        V5StressScenario(
            "slippage_2x",
            strategy,
            replace(costs, slippage_bps=costs.slippage_bps * 2.0),
            "cost",
        ),
        V5StressScenario(
            "execution_delay_plus_1",
            replace(
                strategy,
                execution_delay_sessions=strategy.execution_delay_sessions + 1,
            ),
            costs,
            "delay",
        ),
        V5StressScenario(
            "execution_delay_plus_2",
            replace(
                strategy,
                execution_delay_sessions=strategy.execution_delay_sessions + 2,
            ),
            costs,
            "delay",
        ),
        V5StressScenario(
            "rebalance_10",
            replace(strategy, rebalance_days=10),
            costs,
            "cadence",
        ),
        V5StressScenario(
            "rebalance_20",
            replace(strategy, rebalance_days=20),
            costs,
            "cadence",
        ),
        V5StressScenario(
            "top_n_5",
            replace(strategy, top_n=5),
            costs,
            "breadth",
        ),
        V5StressScenario(
            "top_n_12",
            replace(strategy, top_n=12),
            costs,
            "breadth",
        ),
        V5StressScenario(
            "min_amount_30m",
            replace(strategy, min_amount=max(strategy.min_amount, 30_000_000.0)),
            costs,
            "liquidity",
        ),
        V5StressScenario(
            "min_amount_40m",
            replace(strategy, min_amount=max(strategy.min_amount, 40_000_000.0)),
            costs,
            "liquidity",
        ),
    ]
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate V5 stress scenario name")
    return scenarios


def stress_pass_ratio(rows, *, return_column: str = "total_return") -> float:
    """Share of non-reference stress scenarios that preserve positive return."""
    if rows is None or len(rows) == 0:
        return 0.0
    if return_column not in rows:
        raise ValueError(f"missing stress return column: {return_column}")
    frame = rows.copy()
    if "scenario" in frame:
        frame = frame.loc[frame["scenario"] != "base"]
    if frame.empty:
        return 0.0
    values = frame[return_column]
    return float((values > 0.0).mean())
