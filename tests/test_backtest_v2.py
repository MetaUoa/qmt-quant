import numpy as np
import pandas as pd

from qmt_quant.backtest import rebalance_schedule, run_backtest
from qmt_quant.config import CostConfig, StrategyConfig
from qmt_quant.reference_data import ReferenceData


def make_frame(index, drift):
    close = 10.0 * np.exp(np.arange(len(index)) * drift)
    open_px = close * 1.0005
    return pd.DataFrame(
        {
            "open": open_px,
            "high": np.maximum(open_px, close) * 1.002,
            "low": np.minimum(open_px, close) * 0.998,
            "close": close,
            "volume": 5_000_000.0,
            "amount": 80_000_000.0,
            "preClose": np.r_[np.nan, close[:-1]],
            "suspendFlag": 0.0,
        },
        index=index,
    )


def test_st_stock_is_not_bought_in_strict_pit_mode():
    idx = pd.bdate_range("2018-01-01", "2020-12-31")
    bars = {
        "AAA.SZ": make_frame(idx, 0.0020),
        "BBB.SH": make_frame(idx, 0.0010),
        "000905.SH": make_frame(idx, 0.0008),
    }
    cfg = StrategyConfig(top_n=1, min_amount=1.0, min_price=1.0, max_daily_vol=1.0)
    schedule = rebalance_schedule(idx, cfg)
    basic = pd.DataFrame(
        [
            {"ts_code": "AAA.SZ", "exchange": "SZSE", "list_date": "20100101", "delist_date": None},
            {"ts_code": "BBB.SH", "exchange": "SSE", "list_date": "20100101", "delist_date": None},
        ]
    )
    st_rows = []
    limit_rows = []
    for signal, execution in schedule:
        day = signal.strftime("%Y%m%d")
        st_rows.extend(
            [
                {"trade_date": day, "ts_code": "__NONE__"},
                {"trade_date": day, "ts_code": "AAA.SZ"},
            ]
        )
        exec_day = execution.strftime("%Y%m%d")
        for code in ("AAA.SZ", "BBB.SH"):
            limit_rows.append(
                {
                    "trade_date": exec_day,
                    "ts_code": code,
                    "pre_close": 10.0,
                    "up_limit": 20.0,
                    "down_limit": 5.0,
                }
            )
    ref = ReferenceData(basic, idx, pd.DataFrame(st_rows), pd.DataFrame(limit_rows))
    result = run_backtest(
        bars,
        "000905.SH",
        cfg,
        CostConfig(initial_cash=1_000_000.0),
        reference=ref,
        strict_reference=True,
        limit_reference_bars=bars,
    )
    buys = result.trades[result.trades["side"] == "BUY"] if not result.trades.empty else result.trades
    assert not buys.empty
    assert "AAA.SZ" not in set(buys["code"])
    assert "BBB.SH" in set(buys["code"])
    assert result.metrics["point_in_time_universe"] is True
