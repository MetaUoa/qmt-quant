from __future__ import annotations

from pathlib import Path

import pandas as pd

from qmt_quant.backtest import _panel, _t1_sell_allowed


ROOT = Path(__file__).resolve().parents[1]
BACKTEST = ROOT / "qmt_quant" / "backtest.py"


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


def test_strict_backtest_does_not_fill_unknown_suspension_with_zero():
    text = BACKTEST.read_text(encoding="utf-8")
    assert 'suspend = _panel(stock_bars, "suspendFlag", calendar)' in text
    assert 'suspend = _panel(stock_bars, "suspendFlag", calendar).fillna(0.0)' not in text
    assert "missing_suspend_rows" in text
    assert "if strict_reference:" in text


def test_backtest_reports_daily_limit_model_boundary():
    text = BACKTEST.read_text(encoding="utf-8")
    assert '"t_plus_one_enforced": True' in text
    assert '"intraday_limit_touch_modelled": False' in text
    assert '"limit_model": "open_auction_reference_plus_one_price_daily_fallback"' in text
