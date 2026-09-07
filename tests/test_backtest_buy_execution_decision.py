from __future__ import annotations

import pandas as pd

import qmt_quant.backtest as backtest
import qmt_quant.backtest_buy_execution as buy_execution
from qmt_quant.backtest_buy_execution import evaluate_buy_execution as real_evaluate_buy
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


def _strategy() -> StrategyConfig:
    return StrategyConfig(
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


def test_buy_decision_preserves_fill_then_price_then_cash_scaling(monkeypatch) -> None:
    ts = pd.Timestamp("2025-01-06")
    cost = CostConfig(
        commission_rate=0.00025,
        min_commission=5.0,
        slippage_bps=10.0,
        lot_size=100,
        fill_probability=1.0,
    )

    monkeypatch.setattr(buy_execution, "deterministic_fill", lambda *args, **kwargs: False)

    def should_not_scale(**kwargs):
        raise AssertionError("cash scaling must not run after a rejected deterministic fill")

    monkeypatch.setattr(buy_execution, "affordable_buy_quantity", should_not_scale)
    rejected = real_evaluate_buy(
        requested_shares=300,
        open_price=10.0,
        cash=100_000.0,
        execution_date=ts,
        code="AAA.SZ",
        cost=cost,
    )
    assert not rejected.ready
    assert rejected.quantity == 0
    assert rejected.execution_price is None
    assert rejected.blocked_random_fill == 1

    monkeypatch.setattr(buy_execution, "deterministic_fill", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        buy_execution,
        "affordable_buy_quantity",
        lambda **kwargs: 0,
    )
    unaffordable = real_evaluate_buy(
        requested_shares=300,
        open_price=10.0,
        cash=100.0,
        execution_date=ts,
        code="AAA.SZ",
        cost=cost,
    )
    assert not unaffordable.ready
    assert unaffordable.quantity == 0
    assert unaffordable.execution_price == 10.01
    assert unaffordable.blocked_random_fill == 0

    monkeypatch.setattr(
        buy_execution,
        "affordable_buy_quantity",
        lambda **kwargs: 200,
    )
    ready = real_evaluate_buy(
        requested_shares=300,
        open_price=10.0,
        cash=100_000.0,
        execution_date=ts,
        code="AAA.SZ",
        cost=cost,
    )
    assert ready.ready
    assert ready.quantity == 200
    assert ready.execution_price == 10.01
    assert ready.blocked_random_fill == 0


def test_run_backtest_routes_buys_through_decision_helper(monkeypatch) -> None:
    index = pd.bdate_range("2025-01-02", periods=14)
    bars = {
        "AAA.SZ": _flat_frame(index, 10.0),
        "000905.SH": _flat_frame(index, 100.0),
    }
    score = pd.DataFrame(1.0, index=index, columns=["AAA.SZ"])
    risk_on = pd.Series(True, index=index)
    costs = CostConfig(
        initial_cash=100_000.0,
        commission_rate=0.00025,
        min_commission=5.0,
        slippage_bps=0.0,
        fill_probability=1.0,
    )
    calls: list[tuple[str, int, float, float, pd.Timestamp]] = []

    def tracked_buy_decision(**kwargs):
        calls.append(
            (
                str(kwargs["code"]),
                int(kwargs["requested_shares"]),
                float(kwargs["open_price"]),
                float(kwargs["cash"]),
                pd.Timestamp(kwargs["execution_date"]),
            )
        )
        return real_evaluate_buy(**kwargs)

    monkeypatch.setattr(backtest, "evaluate_buy_execution", tracked_buy_decision)
    result = backtest.run_backtest(
        bars,
        "000905.SH",
        _strategy(),
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert calls
    assert all(code == "AAA.SZ" for code, _, _, _, _ in calls)
    assert all(quantity > 0 for _, quantity, _, _, _ in calls)
    assert "BUY" in set(result.trades["side"])
    assert result.metrics["blocked_random_fill"] == 0
