from __future__ import annotations

import pandas as pd

import qmt_quant.backtest as backtest
from qmt_quant.backtest_execution import build_rebalance_order_plan as real_build_plan
from qmt_quant.config import CostConfig, StrategyConfig


def _flat_frame(index: pd.DatetimeIndex, price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1_000_000.0,
            "amount": 50_000_000.0,
            "preClose": price,
            "suspendFlag": 0.0,
        },
        index=index,
    )


def test_run_backtest_routes_every_rebalance_through_order_plan(monkeypatch) -> None:
    index = pd.bdate_range("2025-01-02", periods=14)
    bars = {
        "AAA.SZ": _flat_frame(index, 10.0),
        "000905.SH": _flat_frame(index, 100.0),
    }
    score = pd.DataFrame(1.0, index=index, columns=["AAA.SZ"])
    risk_on = pd.Series(True, index=index)
    strategy = StrategyConfig(
        mom_short=1,
        mom_mid=1,
        mom_long=1,
        ma_fast=1,
        ma_slow=1,
        vol_window=1,
        amount_window=1,
        benchmark_ma=1,
        benchmark_mom_days=1,
        breadth_ma=1,
        top_n=1,
        rebalance_days=1,
        execution_delay_sessions=1,
        min_price=1.0,
        min_amount=1.0,
        min_momentum=-1.0,
        max_daily_vol=1.0,
        benchmark_mom_floor=-1.0,
        min_listing_sessions=1,
    )
    costs = CostConfig(
        initial_cash=100_000.0,
        commission_rate=0.00025,
        min_commission=5.0,
        slippage_bps=0.0,
        fill_probability=1.0,
    )

    captured = []

    def tracked_plan(**kwargs):
        plan = real_build_plan(**kwargs)
        captured.append(plan)
        return plan

    monkeypatch.setattr(backtest, "build_rebalance_order_plan", tracked_plan)
    result = backtest.run_backtest(
        bars,
        "000905.SH",
        strategy,
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert captured
    assert len(captured) == result.metrics["rebalance_count"]
    assert any(plan.buys for plan in captured)
    assert result.metrics["t_plus_one_enforced"] is True
    assert result.metrics["intraday_limit_touch_modelled"] is False
