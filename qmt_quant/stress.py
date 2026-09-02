from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import CostConfig, StrategyConfig
from .reference_data import ReferenceData


def _subset_universe(
    bars: Dict[str, pd.DataFrame],
    benchmark: str,
    drop_fraction: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    codes = sorted(c for c in bars if c != benchmark)
    if not codes or drop_fraction <= 0:
        return bars
    rng = np.random.default_rng(seed)
    keep_count = max(1, int(round(len(codes) * (1.0 - drop_fraction))))
    keep = set(rng.choice(codes, size=keep_count, replace=False).tolist())
    keep.add(benchmark)
    return {k: v for k, v in bars.items() if k in keep}


def _subset_limit_bars(
    limit_bars: Dict[str, pd.DataFrame] | None,
    keep_codes: set[str],
) -> Dict[str, pd.DataFrame] | None:
    if limit_bars is None:
        return None
    return {k: v for k, v in limit_bars.items() if k in keep_codes}


def _scenario_pass(metrics: dict, *, max_drawdown: float = 0.55) -> bool:
    if not metrics:
        return False
    return (
        float(metrics.get("multiple", 0.0)) > 1.0
        and float(metrics.get("cagr", -1.0)) > 0.0
        and abs(float(metrics.get("max_drawdown", -1.0))) <= max_drawdown
    )


def run_stress_suite(
    bars: Dict[str, pd.DataFrame],
    benchmark: str,
    strategy: StrategyConfig,
    costs: CostConfig,
    *,
    reference: ReferenceData | None = None,
    strict_reference: bool = False,
    limit_reference_bars: Dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    scenarios: list[tuple[str, StrategyConfig, CostConfig, float | None, int | None]] = [
        ("base", strategy, costs, None, None),
        ("commission_x2", strategy, replace(costs, commission_rate=costs.commission_rate * 2), None, None),
        ("slippage_x2", strategy, replace(costs, slippage_bps=costs.slippage_bps * 2), None, None),
        ("slippage_x3", strategy, replace(costs, slippage_bps=costs.slippage_bps * 3), None, None),
        ("execution_delay_2", replace(strategy, execution_delay_sessions=2), costs, None, None),
        ("fill_95pct", strategy, replace(costs, fill_probability=0.95, fill_seed=101), None, None),
        ("fill_90pct", strategy, replace(costs, fill_probability=0.90, fill_seed=202), None, None),
        (
            "momentum_threshold_plus20pct",
            replace(strategy, min_momentum=strategy.min_momentum * 1.20),
            costs,
            None,
            None,
        ),
        (
            "momentum_threshold_minus20pct",
            replace(strategy, min_momentum=max(strategy.min_momentum * 0.80, -0.10)),
            costs,
            None,
            None,
        ),
        (
            "vol_limit_tighter10pct",
            replace(strategy, max_daily_vol=strategy.max_daily_vol * 0.90),
            costs,
            None,
            None,
        ),
        (
            "vol_limit_looser10pct",
            replace(strategy, max_daily_vol=strategy.max_daily_vol * 1.10),
            costs,
            None,
            None,
        ),
        ("universe_drop10_seed1", strategy, costs, 0.10, 1),
        ("universe_drop10_seed2", strategy, costs, 0.10, 2),
        ("universe_drop10_seed3", strategy, costs, 0.10, 3),
    ]

    rows: list[dict] = []
    results: dict[str, BacktestResult] = {}
    for name, cfg, cost, drop_fraction, seed in scenarios:
        scenario_bars = bars
        scenario_limit = limit_reference_bars
        if drop_fraction is not None:
            scenario_bars = _subset_universe(bars, benchmark, drop_fraction, int(seed or 0))
            scenario_limit = _subset_limit_bars(limit_reference_bars, set(scenario_bars))
        result = run_backtest(
            scenario_bars,
            benchmark,
            cfg,
            cost,
            reference=reference,
            strict_reference=strict_reference,
            limit_reference_bars=scenario_limit,
        )
        results[name] = result
        row = {
            "scenario": name,
            "passed": _scenario_pass(result.metrics),
            "multiple": result.metrics.get("multiple"),
            "cagr": result.metrics.get("cagr"),
            "max_drawdown": result.metrics.get("max_drawdown"),
            "sharpe": result.metrics.get("sharpe"),
            "trade_count": result.metrics.get("trade_count"),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, results


def monte_carlo_daily_returns(
    equity: pd.Series,
    *,
    simulations: int = 1000,
    seed: int = 20260902,
) -> dict:
    series = equity.dropna().sort_index()
    rets = series.pct_change().dropna().to_numpy(dtype=float)
    if len(rets) == 0:
        return {}
    rng = np.random.default_rng(seed)
    final_multiples = np.empty(simulations, dtype=float)
    max_drawdowns = np.empty(simulations, dtype=float)
    for i in range(simulations):
        sample = rng.choice(rets, size=len(rets), replace=True)
        curve = np.cumprod(1.0 + sample)
        peak = np.maximum.accumulate(curve)
        dd = curve / peak - 1.0
        final_multiples[i] = curve[-1]
        max_drawdowns[i] = dd.min()
    return {
        "simulations": int(simulations),
        "probability_profitable": float((final_multiples > 1.0).mean()),
        "multiple_p05": float(np.quantile(final_multiples, 0.05)),
        "multiple_p50": float(np.quantile(final_multiples, 0.50)),
        "multiple_p95": float(np.quantile(final_multiples, 0.95)),
        "max_drawdown_p05": float(np.quantile(max_drawdowns, 0.05)),
        "max_drawdown_p50": float(np.quantile(max_drawdowns, 0.50)),
        "max_drawdown_p95": float(np.quantile(max_drawdowns, 0.95)),
    }


def stress_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"scenario_count": 0, "pass_ratio": 0.0}
    non_base = frame.loc[frame["scenario"] != "base"]
    return {
        "scenario_count": int(len(non_base)),
        "passed_scenarios": int(non_base["passed"].astype(bool).sum()),
        "pass_ratio": float(non_base["passed"].astype(bool).mean()) if len(non_base) else 0.0,
        "worst_multiple": float(pd.to_numeric(non_base["multiple"], errors="coerce").min()) if len(non_base) else None,
        "worst_max_drawdown": float(pd.to_numeric(non_base["max_drawdown"], errors="coerce").min()) if len(non_base) else None,
    }
