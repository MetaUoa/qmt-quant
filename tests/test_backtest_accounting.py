from __future__ import annotations

import pandas as pd
import pytest

import qmt_quant.backtest as backtest
from qmt_quant.backtest_execution import settle_buy, settle_sell
from qmt_quant.config import CostConfig, StrategyConfig


def test_sell_settlement_matches_current_inline_accounting_formula() -> None:
    cost = CostConfig(commission_rate=0.001, min_commission=5.0)
    settlement = settle_sell(
        cash=1000.0,
        current_shares=500,
        quantity=200,
        execution_price=10.0,
        cost=cost,
        stamp_tax_rate=0.001,
    )
    notional = 200 * 10.0
    fee = max(5.0, notional * 0.001)
    tax = notional * 0.001
    assert settlement.notional == pytest.approx(notional)
    assert settlement.commission == pytest.approx(fee)
    assert settlement.stamp_tax == pytest.approx(tax)
    assert settlement.ending_cash == pytest.approx(1000.0 + notional - fee - tax)
    assert settlement.ending_shares == 300


def test_buy_settlement_matches_current_inline_accounting_formula() -> None:
    cost = CostConfig(commission_rate=0.001, min_commission=5.0)
    settlement = settle_buy(
        cash=5000.0,
        current_shares=100,
        quantity=200,
        execution_price=10.0,
        cost=cost,
    )
    notional = 200 * 10.0
    fee = max(5.0, notional * 0.001)
    assert settlement.notional == pytest.approx(notional)
    assert settlement.commission == pytest.approx(fee)
    assert settlement.ending_cash == pytest.approx(5000.0 - notional - fee)
    assert settlement.ending_shares == 300


def test_accounting_helpers_do_not_apply_execution_or_t1_policy() -> None:
    cost = CostConfig(commission_rate=0.0, min_commission=0.0)
    sell = settle_sell(
        cash=0.0,
        current_shares=100,
        quantity=100,
        execution_price=8.0,
        cost=cost,
        stamp_tax_rate=0.0005,
    )
    buy = settle_buy(
        cash=1000.0,
        current_shares=0,
        quantity=100,
        execution_price=8.0,
        cost=cost,
    )
    assert sell.ending_shares == 0
    assert buy.ending_shares == 100


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


def test_run_backtest_routes_filled_orders_through_accounting_helpers(monkeypatch) -> None:
    index = pd.bdate_range("2025-01-02", periods=14)
    bars = {
        "AAA.SZ": _flat_frame(index, 10.0),
        "BBB.SZ": _flat_frame(index, 20.0),
        "000905.SH": _flat_frame(index, 100.0),
    }
    score = pd.DataFrame(index=index, columns=["AAA.SZ", "BBB.SZ"], dtype=float)
    for i, ts in enumerate(index):
        if i % 2 == 0:
            score.loc[ts] = [2.0, 1.0]
        else:
            score.loc[ts] = [1.0, 2.0]
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

    calls = {"buy": 0, "sell": 0}
    real_buy = backtest.settle_buy
    real_sell = backtest.settle_sell

    def tracked_buy(**kwargs):
        calls["buy"] += 1
        return real_buy(**kwargs)

    def tracked_sell(**kwargs):
        calls["sell"] += 1
        return real_sell(**kwargs)

    monkeypatch.setattr(backtest, "settle_buy", tracked_buy)
    monkeypatch.setattr(backtest, "settle_sell", tracked_sell)

    result = backtest.run_backtest(
        bars,
        "000905.SH",
        strategy,
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert calls["buy"] > 0
    assert calls["sell"] > 0
    assert not result.trades.empty
    assert set(result.trades["side"]) == {"BUY", "SELL"}
