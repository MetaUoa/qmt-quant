from __future__ import annotations

import numpy as np
import pandas as pd

from qmt_quant.backtest import _panel, _t1_sell_allowed
from qmt_quant.backtest_execution import TradabilityGuard
from qmt_quant.backtest_reporting import BacktestDiagnostics, assemble_backtest_metrics


def test_t1_sellability_is_explicit_by_calendar_date():
    assert _t1_sell_allowed(None, pd.Timestamp("2025-01-03")) is True
    assert _t1_sell_allowed(pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")) is True
    assert _t1_sell_allowed(pd.Timestamp("2025-01-03 09:31"), pd.Timestamp("2025-01-03 14:55")) is False
    assert _t1_sell_allowed(pd.Timestamp("2025-01-04"), pd.Timestamp("2025-01-03")) is False


def test_panel_concat_preserves_calendar_alignment_and_symbol_order():
    calendar = pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    bars = {
        "000001.SZ": pd.DataFrame(
            {"close": [10.0, 10.2]},
            index=pd.to_datetime(["2025-01-02", "2025-01-06"]),
        ),
        "600000.SH": pd.DataFrame(
            {"close": [8.0, 8.1, 8.2]},
            index=calendar,
        ),
    }
    panel = _panel(bars, "close", calendar)
    assert list(panel.columns) == ["000001.SZ", "600000.SH"]
    assert panel.index.equals(calendar)
    assert pd.isna(panel.loc[pd.Timestamp("2025-01-03"), "000001.SZ"])
    assert panel.loc[pd.Timestamp("2025-01-06"), "600000.SH"] == 8.2


def test_non_strict_unknown_suspension_uses_open_quote_fallback():
    dates = pd.DatetimeIndex(["2025-01-03"])
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
        strict_reference=False,
        raw_limit_reference_supplied=True,
        limit_tolerance=0.001,
    )
    assert guard.is_halted(dates[0], "000001.SZ") is False
    assert guard.missing_suspend_rows == 0


def test_backtest_reports_daily_limit_model_boundary():
    metrics = assemble_backtest_metrics(
        {},
        BacktestDiagnostics(
            trade_count=0,
            rebalance_count=0,
            initial_cash=1_000_000.0,
            blocked_st_candidates=0,
            blocked_limit_buys=0,
            blocked_limit_sells=0,
            blocked_suspended=0,
            blocked_t1_sells=0,
            missing_suspend_rows=0,
            missing_limit_rows=0,
            missing_st_dates=0,
            missing_limit_dates=0,
            point_in_time_universe=True,
            strict_reference=True,
            raw_limit_reference=True,
            blocked_random_fill=0,
            execution_delay_sessions=1,
            fill_probability=1.0,
            average_market_breadth=0.5,
        ),
    )
    assert metrics["t_plus_one_enforced"] is True
    assert metrics["intraday_limit_touch_modelled"] is False
    assert metrics["limit_model"] == "open_auction_reference_plus_one_price_daily_fallback"
