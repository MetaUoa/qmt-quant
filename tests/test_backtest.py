import numpy as np
import pandas as pd

from qmt_quant.backtest import run_backtest
from qmt_quant.config import CostConfig, StrategyConfig


def make_frame(index, seed, drift):
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0.0, 0.012, len(index))
    close = 10.0 * np.exp(np.cumsum(returns))
    open_px = close * (1.0 + rng.normal(0.0, 0.002, len(index)))
    return pd.DataFrame(
        {
            "open": open_px,
            "high": np.maximum(open_px, close) * 1.01,
            "low": np.minimum(open_px, close) * 0.99,
            "close": close,
            "volume": 5_000_000.0,
            "amount": 80_000_000.0,
            "preClose": np.r_[np.nan, close[:-1]],
            "suspendFlag": 0.0,
        },
        index=index,
    )


def test_backtest_runs_and_signal_precedes_trade():
    idx = pd.bdate_range("2018-01-01", "2021-12-31")
    bars = {
        "AAA.SZ": make_frame(idx, 1, 0.0012),
        "BBB.SZ": make_frame(idx, 2, 0.0005),
        "CCC.SH": make_frame(idx, 3, -0.0001),
        "000905.SH": make_frame(idx, 9, 0.0004),
    }
    cfg = StrategyConfig(top_n=2, min_amount=1.0, min_price=1.0, max_daily_vol=1.0)
    result = run_backtest(bars, "000905.SH", cfg, CostConfig(initial_cash=1_000_000.0))
    assert not result.equity.empty
    assert result.metrics["ending_equity"] > 0
    if not result.trades.empty:
        assert (pd.to_datetime(result.trades["signal_date"]) < pd.to_datetime(result.trades["date"])).all()
