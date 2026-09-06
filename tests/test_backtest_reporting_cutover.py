from __future__ import annotations

import pandas as pd

import qmt_quant.backtest as backtest
from qmt_quant.backtest_reporting import assemble_backtest_metrics as real_assemble
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


def test_run_backtest_routes_public_metrics_through_reporting_assembler(monkeypatch) -> None:
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

    def tracked_assemble(base_metrics, diagnostics):
        captured.append(diagnostics)
        return real_assemble(base_metrics, diagnostics)

    monkeypatch.setattr(backtest, "assemble_backtest_metrics", tracked_assemble)
    result = backtest.run_backtest(
        bars,
        "000905.SH",
        strategy,
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert len(captured) == 1
    diagnostics = captured[0]
    assert diagnostics.trade_count == len(result.trades)
    assert diagnostics.rebalance_count == result.metrics["rebalance_count"]
    assert result.metrics["t_plus_one_enforced"] is True
    assert result.metrics["limit_model"] == "open_auction_reference_plus_one_price_daily_fallback"
    assert result.metrics["intraday_limit_touch_modelled"] is False
    assert result.metrics["score_override"] is True
    assert result.metrics["risk_on_override"] is True
