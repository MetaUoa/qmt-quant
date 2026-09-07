from __future__ import annotations

import pandas as pd

import qmt_quant.backtest as backtest
from qmt_quant.backtest_execution import (
    apply_buy_position_mutation as real_apply_buy,
    apply_sell_position_mutation as real_apply_sell,
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


def test_position_mutation_helpers_preserve_existing_lifecycle() -> None:
    positions = {"AAA.SZ": 300, "BBB.SZ": 100}
    last_buy = {
        "AAA.SZ": pd.Timestamp("2025-01-02"),
        "BBB.SZ": pd.Timestamp("2025-01-03"),
    }

    real_apply_sell(
        positions=positions,
        last_buy_date=last_buy,
        code="AAA.SZ",
        ending_shares=100,
    )
    assert positions == {"AAA.SZ": 100, "BBB.SZ": 100}
    assert last_buy["AAA.SZ"] == pd.Timestamp("2025-01-02")

    real_apply_sell(
        positions=positions,
        last_buy_date=last_buy,
        code="AAA.SZ",
        ending_shares=0,
    )
    assert positions == {"BBB.SZ": 100}
    assert "AAA.SZ" not in last_buy

    real_apply_buy(
        positions=positions,
        last_buy_date=last_buy,
        code="CCC.SZ",
        ending_shares=200,
        execution_date=pd.Timestamp("2025-01-06 14:30:00"),
    )
    assert positions == {"BBB.SZ": 100, "CCC.SZ": 200}
    assert last_buy["CCC.SZ"] == pd.Timestamp("2025-01-06")


def test_run_backtest_routes_buy_and_sell_state_through_mutation_helpers(monkeypatch) -> None:
    index = pd.bdate_range("2025-01-02", periods=14)
    bars = {
        "AAA.SZ": _flat_frame(index, 10.0),
        "000905.SH": _flat_frame(index, 100.0),
    }
    score = pd.DataFrame(1.0, index=index, columns=["AAA.SZ"])
    risk_on = pd.Series(False, index=index)
    # With all lookbacks/min-listing set to 1, StrategyConfig.warmup is 6 and
    # the first delayed signal is index 5. Keep the early window risk-on long
    # enough to guarantee a BUY, then turn it off to force the later SELL path.
    risk_on.iloc[:8] = True
    costs = CostConfig(
        initial_cash=100_000.0,
        commission_rate=0.00025,
        min_commission=5.0,
        slippage_bps=0.0,
        fill_probability=1.0,
    )
    calls: list[tuple[str, str, int]] = []

    def tracked_buy(**kwargs) -> None:
        calls.append(("BUY", str(kwargs["code"]), int(kwargs["ending_shares"])))
        real_apply_buy(**kwargs)

    def tracked_sell(**kwargs) -> None:
        calls.append(("SELL", str(kwargs["code"]), int(kwargs["ending_shares"])))
        real_apply_sell(**kwargs)

    monkeypatch.setattr(backtest, "apply_buy_position_mutation", tracked_buy)
    monkeypatch.setattr(backtest, "apply_sell_position_mutation", tracked_sell)
    result = backtest.run_backtest(
        bars,
        "000905.SH",
        _strategy(),
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert any(side == "BUY" for side, _, _ in calls)
    assert any(side == "SELL" for side, _, _ in calls)
    assert set(result.trades["side"]) == {"BUY", "SELL"}
    assert result.metrics["t_plus_one_enforced"] is True
