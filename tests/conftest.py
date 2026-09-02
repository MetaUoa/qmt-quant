from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qmt_quant.config import CostConfig, StrategyConfig


@pytest.fixture
def trading_index() -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", "2022-12-30")


def make_bar_frame(
    index: pd.DatetimeIndex,
    *,
    drift: float = 0.0010,
    start_price: float = 10.0,
    seed: int = 1,
    noise: float = 0.0020,
    amount: float = 80_000_000.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0.0, noise, len(index))
    close = start_price * np.exp(np.cumsum(returns))
    open_px = close * (1.0 + rng.normal(0.0, 0.0005, len(index)))
    high = np.maximum(open_px, close) * 1.002
    low = np.minimum(open_px, close) * 0.998
    return pd.DataFrame(
        {
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": 5_000_000.0,
            "amount": amount,
            "preClose": np.r_[np.nan, close[:-1]],
            "suspendFlag": 0.0,
        },
        index=index,
    )


@pytest.fixture
def synthetic_bars(trading_index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    idx = trading_index
    return {
        "AAA.SZ": make_bar_frame(idx, drift=0.0015, seed=1),
        "BBB.SH": make_bar_frame(idx, drift=0.0009, seed=2),
        "CCC.SZ": make_bar_frame(idx, drift=0.0003, seed=3),
        "000905.SH": make_bar_frame(idx, drift=0.0007, seed=9, noise=0.0010),
    }


@pytest.fixture
def permissive_strategy() -> StrategyConfig:
    return StrategyConfig(
        top_n=2,
        rebalance_days=5,
        min_price=1.0,
        min_amount=1.0,
        min_momentum=-1.0,
        max_daily_vol=1.0,
        min_listing_sessions=0,
    )


@pytest.fixture
def low_costs() -> CostConfig:
    return CostConfig(
        initial_cash=1_000_000.0,
        commission_rate=0.00025,
        min_commission=5.0,
        slippage_bps=5.0,
        lot_size=100,
    )
