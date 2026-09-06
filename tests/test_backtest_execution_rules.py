from __future__ import annotations

import numpy as np
import pandas as pd

from qmt_quant.backtest_execution import (
    TradabilityGuard,
    affordable_buy_quantity,
    commission,
    deterministic_fill,
    equal_weight_target_shares,
    mark_portfolio_value,
)
from qmt_quant.config import CostConfig


def test_deterministic_fill_and_commission_preserve_cost_semantics():
    cost = CostConfig(
        fill_probability=0.5,
        fill_seed=123,
        commission_rate=0.001,
        min_commission=5.0,
    )
    ts = pd.Timestamp("2024-01-02")
    first = deterministic_fill(cost, ts, "000001.SZ", "BUY")
    assert deterministic_fill(cost, ts, "000001.SZ", "BUY") is first
    assert commission(cost, 1000.0) == 5.0
    assert commission(cost, 10000.0) == 10.0
    assert commission(cost, 0.0) == 0.0


def test_equal_weight_target_shares_matches_current_lot_flooring():
    ts = pd.Timestamp("2024-01-02")
    open_px = pd.DataFrame(
        {"000001.SZ": [10.0], "000002.SZ": [20.0]},
        index=pd.DatetimeIndex([ts]),
    )
    desired = equal_weight_target_shares(
        selected=["000001.SZ", "000002.SZ"],
        open_px=open_px,
        execution_date=ts,
        portfolio_value=100_000.0,
        exposure=1.0,
        slippage_bps=10.0,
        lot_size=100,
    )
    assert desired == {"000001.SZ": 4900, "000002.SZ": 2400}
    assert equal_weight_target_shares(
        selected=[],
        open_px=open_px,
        execution_date=ts,
        portfolio_value=100_000.0,
        exposure=1.0,
        slippage_bps=10.0,
        lot_size=100,
    ) == {}


def test_affordable_buy_quantity_scales_down_by_board_lots_with_commission():
    cost = CostConfig(lot_size=100, commission_rate=0.001, min_commission=5.0)
    assert affordable_buy_quantity(
        requested_shares=1000,
        execution_price=10.0,
        cash=10_020.0,
        cost=cost,
    ) == 1000
    assert affordable_buy_quantity(
        requested_shares=1000,
        execution_price=10.0,
        cash=9_500.0,
        cost=cost,
    ) == 900
    assert affordable_buy_quantity(
        requested_shares=100,
        execution_price=10.0,
        cash=900.0,
        cost=cost,
    ) == 0


def test_strict_unknown_suspension_is_fail_closed_and_counted():
    dates = pd.DatetimeIndex(["2024-01-02"])
    open_px = pd.DataFrame({"000001.SZ": [10.0]}, index=dates)
    guard = TradabilityGuard(
        calendar=dates,
        open_px=open_px,
        high_px=open_px,
        low_px=open_px,
        close_px=open_px,
        suspend=pd.DataFrame({"000001.SZ": [np.nan]}, index=dates),
        limit_open_px=open_px,
        limit_preclose_px=pd.DataFrame({"000001.SZ": [9.5]}, index=dates),
        reference=None,
        strict_reference=True,
        raw_limit_reference_supplied=True,
        limit_tolerance=0.001,
    )
    assert guard.is_halted(dates[0], "000001.SZ") is True
    assert guard.missing_suspend_rows == 1


def test_mark_portfolio_value_uses_last_close_when_current_quote_missing():
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    close_px = pd.DataFrame({"000001.SZ": [10.0, np.nan]}, index=dates)
    value = mark_portfolio_value(
        cash=100.0,
        positions={"000001.SZ": 100},
        matrix=close_px,
        close_px=close_px,
        calendar=dates,
        index=1,
        reference=None,
    )
    assert value == 1100.0
