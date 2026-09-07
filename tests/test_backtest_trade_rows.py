from __future__ import annotations

import pandas as pd

import qmt_quant.backtest as backtest
from qmt_quant.backtest_execution import BuySettlement, SellSettlement
from qmt_quant.backtest_trades import (
    build_buy_trade_row as real_build_buy_row,
    build_sell_trade_row as real_build_sell_row,
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


def test_trade_row_builders_lock_public_payload_and_key_order() -> None:
    execution_date = pd.Timestamp("2025-01-06")
    signal_date = pd.Timestamp("2025-01-03")
    sell = SellSettlement(
        ending_cash=101_000.0,
        ending_shares=0,
        notional=1_000.0,
        commission=5.0,
        stamp_tax=0.5,
    )
    buy = BuySettlement(
        ending_cash=98_995.0,
        ending_shares=100,
        notional=1_000.0,
        commission=5.0,
    )

    sell_row = real_build_sell_row(
        execution_date=execution_date,
        signal_date=signal_date,
        code="AAA.SZ",
        shares=100,
        execution_price=10.0,
        settlement=sell,
    )
    buy_row = real_build_buy_row(
        execution_date=execution_date,
        signal_date=signal_date,
        code="AAA.SZ",
        shares=100,
        execution_price=10.0,
        settlement=buy,
    )

    expected_keys = [
        "date",
        "code",
        "side",
        "shares",
        "price",
        "notional",
        "commission",
        "stamp_tax",
        "signal_date",
    ]
    assert list(sell_row) == expected_keys
    assert sell_row == {
        "date": execution_date,
        "code": "AAA.SZ",
        "side": "SELL",
        "shares": 100,
        "price": 10.0,
        "notional": 1_000.0,
        "commission": 5.0,
        "stamp_tax": 0.5,
        "signal_date": signal_date,
    }
    assert list(buy_row) == expected_keys
    assert buy_row == {
        "date": execution_date,
        "code": "AAA.SZ",
        "side": "BUY",
        "shares": 100,
        "price": 10.0,
        "notional": 1_000.0,
        "commission": 5.0,
        "stamp_tax": 0.0,
        "signal_date": signal_date,
    }


def test_run_backtest_routes_trade_records_through_builders(monkeypatch) -> None:
    index = pd.bdate_range("2025-01-02", periods=14)
    bars = {
        "AAA.SZ": _flat_frame(index, 10.0),
        "000905.SH": _flat_frame(index, 100.0),
    }
    score = pd.DataFrame(1.0, index=index, columns=["AAA.SZ"])
    risk_on = pd.Series(False, index=index)
    risk_on.iloc[:8] = True
    costs = CostConfig(
        initial_cash=100_000.0,
        commission_rate=0.00025,
        min_commission=5.0,
        slippage_bps=0.0,
        fill_probability=1.0,
    )
    calls: list[str] = []

    def tracked_buy(**kwargs):
        calls.append("BUY")
        return real_build_buy_row(**kwargs)

    def tracked_sell(**kwargs):
        calls.append("SELL")
        return real_build_sell_row(**kwargs)

    monkeypatch.setattr(backtest, "build_buy_trade_row", tracked_buy)
    monkeypatch.setattr(backtest, "build_sell_trade_row", tracked_sell)
    result = backtest.run_backtest(
        bars,
        "000905.SH",
        _strategy(),
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert "BUY" in calls
    assert "SELL" in calls
    assert list(result.trades.columns) == [
        "date",
        "code",
        "side",
        "shares",
        "price",
        "notional",
        "commission",
        "stamp_tax",
        "signal_date",
    ]
    assert set(result.trades["side"]) == {"BUY", "SELL"}
