from __future__ import annotations

import pandas as pd

import qmt_quant.backtest as backtest
from qmt_quant.backtest_execution import (
    affordable_buy_quantity as real_affordable_buy_quantity,
    deterministic_fill as real_deterministic_fill,
    equal_weight_target_shares as real_equal_weight_target_shares,
)
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


def test_backtest_routes_target_cash_sizing_and_fill_through_pure_helpers(monkeypatch) -> None:
    index = pd.bdate_range("2025-01-02", periods=18)
    bars = {
        "AAA.SZ": _flat_frame(index, 10.0),
        "BBB.SZ": _flat_frame(index, 20.0),
        "000905.SH": _flat_frame(index, 100.0),
    }
    score = pd.DataFrame(index=index, columns=["AAA.SZ", "BBB.SZ"], dtype=float)
    score.loc[:, "AAA.SZ"] = 2.0
    score.loc[:, "BBB.SZ"] = 1.0
    score.loc[index[10]:, "AAA.SZ"] = 1.0
    score.loc[index[10]:, "BBB.SZ"] = 2.0
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

    events: list[str] = []

    def tracked_target(**kwargs):
        events.append("target")
        return real_equal_weight_target_shares(**kwargs)

    def tracked_affordable(**kwargs):
        events.append("affordable")
        return real_affordable_buy_quantity(**kwargs)

    def tracked_fill(cost, ts, code, side):
        events.append(f"fill:{side}")
        return real_deterministic_fill(cost, ts, code, side)

    monkeypatch.setattr(backtest, "equal_weight_target_shares", tracked_target)
    monkeypatch.setattr(backtest, "affordable_buy_quantity", tracked_affordable)
    monkeypatch.setattr(backtest, "deterministic_fill", tracked_fill)

    result = backtest.run_backtest(
        bars,
        "000905.SH",
        strategy,
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert "target" in events
    assert "affordable" in events
    assert "fill:SELL" in events
    assert "fill:BUY" in events

    sides_by_date = result.trades.groupby("date", sort=False)["side"].apply(list)
    switch_days = [sides for sides in sides_by_date if "SELL" in sides and "BUY" in sides]
    assert switch_days
    assert all(sides.index("SELL") < sides.index("BUY") for sides in switch_days)
