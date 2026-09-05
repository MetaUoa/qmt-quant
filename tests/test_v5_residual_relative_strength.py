from __future__ import annotations

import numpy as np
import pandas as pd

from qmt_quant.factors import V5FactorConfig, build_v5_raw_factors
from qmt_quant.residual_strength import residual_relative_strength


def _prices_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.cumprod(1.0 + returns)


def test_residual_strength_differs_from_plain_momentum_when_betas_differ():
    dates = pd.bdate_range("2020-01-01", periods=260)
    market_ret = np.linspace(-0.003, 0.004, len(dates))
    stock_a = market_ret * 0.5 + 0.0005
    stock_b = market_ret * 1.8 + 0.0005
    close = pd.DataFrame(
        {
            "A": _prices_from_returns(stock_a),
            "B": _prices_from_returns(stock_b),
        },
        index=dates,
    )
    benchmark = pd.Series(_prices_from_returns(market_ret), index=dates)
    residual = residual_relative_strength(
        close,
        benchmark,
        lookback=60,
        skip_recent=5,
        beta_window=120,
        beta_min_periods=60,
    )
    plain = close.shift(5).div(close.shift(60)).sub(1.0)
    ts = dates[-1]
    assert residual.loc[ts].notna().all()
    assert not np.allclose(residual.loc[ts].to_numpy(), plain.loc[ts].to_numpy())


def test_future_price_change_does_not_change_past_residual_strength():
    dates = pd.bdate_range("2020-01-01", periods=260)
    market_ret = np.sin(np.arange(len(dates)) / 20.0) * 0.002
    close = pd.DataFrame(
        {
            "A": _prices_from_returns(market_ret * 0.8 + 0.0002),
            "B": _prices_from_returns(market_ret * 1.2 - 0.0001),
        },
        index=dates,
    )
    benchmark = pd.Series(_prices_from_returns(market_ret), index=dates)
    before = residual_relative_strength(close, benchmark)
    mutated = close.copy()
    mutated.loc[dates[-10]:, "A"] *= 4.0
    after = residual_relative_strength(mutated, benchmark)
    cutoff = dates[-20]
    pd.testing.assert_series_equal(before.loc[cutoff], after.loc[cutoff])


def test_v5_factor_generator_exposes_nonredundant_residual_strength():
    dates = pd.bdate_range("2020-01-01", periods=260)
    market_ret = np.cos(np.arange(len(dates)) / 25.0) * 0.002
    close = pd.DataFrame(
        {
            "A": _prices_from_returns(market_ret * 0.5 + 0.0004),
            "B": _prices_from_returns(market_ret * 1.7 - 0.0002),
        },
        index=dates,
    )
    amount = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)
    benchmark = pd.Series(_prices_from_returns(market_ret), index=dates)
    factors = build_v5_raw_factors(close, amount, benchmark, V5FactorConfig())
    assert "residual_relative_strength_60_5" in factors
    ts = dates[-1]
    assert not factors["residual_relative_strength_60_5"].loc[ts].equals(
        factors["momentum_60_5"].loc[ts]
    )
