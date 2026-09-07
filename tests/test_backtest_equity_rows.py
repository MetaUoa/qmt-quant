from __future__ import annotations

import pandas as pd

import qmt_quant.backtest as backtest
from qmt_quant.backtest_equity import build_equity_row as real_build_equity_row
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


def test_build_equity_row_preserves_public_payload_types_and_values() -> None:
    ts = pd.Timestamp("2025-01-06")
    row = real_build_equity_row(
        date=ts,
        equity=100_123.45,
        cash=12_345.67,
        position_count=3,
        risk_on=True,
    )

    assert row == {
        "date": ts,
        "equity": 100_123.45,
        "cash": 12_345.67,
        "positions": 3,
        "risk_on": True,
    }


def test_run_backtest_routes_every_daily_row_through_equity_builder(monkeypatch) -> None:
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
    rows: list[dict[str, object]] = []

    def tracked_equity_row(**kwargs):
        row = real_build_equity_row(**kwargs)
        rows.append(row)
        return row

    monkeypatch.setattr(backtest, "build_equity_row", tracked_equity_row)
    result = backtest.run_backtest(
        bars,
        "000905.SH",
        _strategy(),
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert len(rows) == len(index)
    expected = pd.DataFrame(rows).set_index("date")
    pd.testing.assert_frame_equal(result.equity, expected)
    assert rows[0]["risk_on"] is False
    assert any(bool(row["risk_on"]) for row in rows[7:])
