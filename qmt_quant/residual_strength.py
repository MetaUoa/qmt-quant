from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_market_beta(
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    *,
    window: int = 120,
    skip_recent: int = 5,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Estimate date-local trailing market beta using only observations available by t-skip."""
    window = int(window)
    skip_recent = int(skip_recent)
    if window <= 1 or skip_recent < 0:
        raise ValueError("window must exceed 1 and skip_recent must be non-negative")
    minimum = int(min_periods) if min_periods is not None else max(window // 2, 20)
    if minimum <= 1 or minimum > window:
        raise ValueError("min_periods must be in [2, window]")

    stock_ret = close.sort_index().astype(float).pct_change(fill_method=None).shift(skip_recent)
    benchmark_ret = (
        benchmark_close.reindex(stock_ret.index).astype(float).pct_change(fill_method=None).shift(skip_recent)
    )
    benchmark_var = benchmark_ret.rolling(window, min_periods=minimum).var()
    beta = pd.DataFrame(index=stock_ret.index, columns=stock_ret.columns, dtype=float)
    for column in stock_ret.columns:
        covariance = stock_ret[column].rolling(window, min_periods=minimum).cov(benchmark_ret)
        beta[column] = covariance.div(benchmark_var.replace(0.0, np.nan))
    return beta


def residual_relative_strength(
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    *,
    lookback: int = 60,
    skip_recent: int = 5,
    beta_window: int = 120,
    beta_min_periods: int | None = None,
) -> pd.DataFrame:
    """Market-beta-adjusted momentum, unlike subtracting one benchmark scalar.

    For each stock/date:
      residual strength = stock skip-recent momentum - beta * benchmark momentum.

    Beta is estimated from trailing daily returns ending at t-skip_recent, so the
    factor is point-in-time and does not consume future validation data.
    """
    lookback = int(lookback)
    skip_recent = int(skip_recent)
    if lookback <= skip_recent or skip_recent < 0:
        raise ValueError("lookback must be greater than non-negative skip_recent")
    ordered = close.sort_index().astype(float)
    benchmark = benchmark_close.reindex(ordered.index).astype(float)
    stock_momentum = ordered.shift(skip_recent).div(ordered.shift(lookback)).sub(1.0)
    benchmark_momentum = benchmark.shift(skip_recent).div(benchmark.shift(lookback)).sub(1.0)
    beta = rolling_market_beta(
        ordered,
        benchmark,
        window=beta_window,
        skip_recent=skip_recent,
        min_periods=beta_min_periods,
    )
    return stock_momentum.sub(beta.mul(benchmark_momentum, axis=0))
