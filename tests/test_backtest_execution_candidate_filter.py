from __future__ import annotations

from typing import cast

import pandas as pd

import qmt_quant.backtest as backtest
from qmt_quant.backtest_execution import (
    TradabilityGuard,
    filter_execution_candidates as real_filter,
)
from qmt_quant.config import CostConfig, StrategyConfig


class _FakeGuard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def is_halted(self, ts: pd.Timestamp, code: str) -> bool:
        self.calls.append(("halt", code, None))
        return code == "HALT"

    def limit_blocked(self, ts: pd.Timestamp, code: str, side: str) -> bool:
        self.calls.append(("limit", code, side))
        return code == "LIMIT"


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


def test_filter_execution_candidates_preserves_short_circuit_and_rank_order() -> None:
    guard = _FakeGuard()
    ts = pd.Timestamp("2025-01-06")

    result = real_filter(
        candidates=("HALT", "LIMIT", "OK"),
        guard=cast(TradabilityGuard, guard),
        execution_date=ts,
    )

    assert result.selected == ("OK",)
    assert result.blocked_suspended == 1
    assert result.blocked_limit_buys == 1
    assert guard.calls == [
        ("halt", "HALT", None),
        ("halt", "LIMIT", None),
        ("limit", "LIMIT", "BUY"),
        ("halt", "OK", None),
        ("limit", "OK", "BUY"),
    ]


def test_run_backtest_routes_rebalances_through_execution_candidate_filter(monkeypatch) -> None:
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
    captured = []

    def tracked_filter(**kwargs):
        result = real_filter(**kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(backtest, "filter_execution_candidates", tracked_filter)
    result = backtest.run_backtest(
        bars,
        "000905.SH",
        _strategy(),
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert captured
    assert len(captured) == result.metrics["rebalance_count"]
    assert all(item.selected == ("AAA.SZ",) for item in captured)
    assert result.metrics["blocked_suspended"] == 0
    assert result.metrics["blocked_limit_buys"] == 0
