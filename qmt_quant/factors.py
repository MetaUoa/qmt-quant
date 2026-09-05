from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V5FactorConfig:
    """Configuration for the V5 cross-sectional alpha research layer.

    Every factor is built only from information available on or before the factor
    timestamp. Forward returns belong in diagnostics, never in this module.
    """

    skip_recent: int = 5
    momentum_short: int = 20
    momentum_mid: int = 60
    momentum_long: int = 120
    trend_fast: int = 20
    trend_slow: int = 60
    vol_window: int = 20
    downside_window: int = 60
    amount_window: int = 20
    amount_stability_window: int = 20
    reversal_window: int = 5
    winsor_lower: float = 0.025
    winsor_upper: float = 0.975

    @property
    def warmup(self) -> int:
        return max(
            self.momentum_long,
            self.trend_slow,
            self.vol_window,
            self.downside_window,
            self.amount_window,
            self.amount_stability_window,
        ) + self.skip_recent + 5


def cross_sectional_winsorize(
    frame: pd.DataFrame,
    lower: float = 0.025,
    upper: float = 0.975,
) -> pd.DataFrame:
    """Winsorize each date independently without carrying information across time."""
    if not 0.0 <= float(lower) < float(upper) <= 1.0:
        raise ValueError("winsor bounds must satisfy 0 <= lower < upper <= 1")
    if frame.empty:
        return frame.copy()
    lo = frame.quantile(float(lower), axis=1, numeric_only=True)
    hi = frame.quantile(float(upper), axis=1, numeric_only=True)
    return frame.clip(lower=lo, upper=hi, axis=0)


def cross_sectional_rank(frame: pd.DataFrame, *, centered: bool = True) -> pd.DataFrame:
    """Percentile-rank symbols independently on each date.

    Centered ranks are in [-1, 1], which makes factor weights comparable and avoids
    using future distributional information for normalization.
    """
    ranked = frame.rank(axis=1, method="average", pct=True, na_option="keep")
    return ranked * 2.0 - 1.0 if centered else ranked


def normalize_factor(
    frame: pd.DataFrame,
    *,
    lower: float = 0.025,
    upper: float = 0.975,
) -> pd.DataFrame:
    return cross_sectional_rank(cross_sectional_winsorize(frame, lower, upper))


def _skip_return(close: pd.DataFrame, lookback: int, skip_recent: int) -> pd.DataFrame:
    lookback = int(lookback)
    skip_recent = int(skip_recent)
    if lookback <= skip_recent or skip_recent < 0:
        raise ValueError("lookback must be greater than non-negative skip_recent")
    return close.shift(skip_recent).div(close.shift(lookback)).sub(1.0)


def iter_v5_raw_factors(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    config: V5FactorConfig | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield V5 factors one at a time to bound memory on the full A-share market."""
    cfg = config or V5FactorConfig()
    close = close.sort_index().astype(float)
    amount = amount.reindex_like(close).astype(float)

    mom_short = _skip_return(close, cfg.momentum_short, cfg.skip_recent)
    yield "momentum_20_5", mom_short
    del mom_short

    mom_mid = _skip_return(close, cfg.momentum_mid, cfg.skip_recent)
    yield "momentum_60_5", mom_mid
    if benchmark_close is not None:
        benchmark = benchmark_close.reindex(close.index).astype(float)
        benchmark_mom = _skip_return(
            benchmark.to_frame("benchmark"),
            cfg.momentum_mid,
            cfg.skip_recent,
        )["benchmark"]
        yield "relative_strength_60_5", mom_mid.sub(benchmark_mom, axis=0)
        del benchmark, benchmark_mom
    del mom_mid

    mom_long = _skip_return(close, cfg.momentum_long, cfg.skip_recent)
    yield "momentum_120_5", mom_long
    del mom_long

    fast_ma = close.rolling(cfg.trend_fast, min_periods=cfg.trend_fast).mean()
    slow_ma = close.rolling(cfg.trend_slow, min_periods=cfg.trend_slow).mean()
    yield "trend_quality", fast_ma.div(slow_ma).sub(1.0)
    del fast_ma, slow_ma

    daily_ret = close.pct_change(fill_method=None)
    trend_persistence = (
        daily_ret.gt(0.0)
        .rolling(cfg.trend_slow, min_periods=cfg.trend_slow)
        .mean()
        .sub(0.5)
    )
    yield "trend_persistence", trend_persistence
    del trend_persistence

    realized_vol = daily_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std()
    yield "low_volatility", -realized_vol
    del realized_vol

    downside = daily_ret.clip(upper=0.0)
    downside_vol = (
        downside.pow(2)
        .rolling(cfg.downside_window, min_periods=cfg.downside_window)
        .mean()
        .pow(0.5)
    )
    yield "low_downside_risk", -downside_vol
    del downside, downside_vol, daily_ret

    avg_amount = amount.rolling(cfg.amount_window, min_periods=cfg.amount_window).mean()
    yield "liquidity", np.log1p(avg_amount.clip(lower=0.0))

    amount_std = amount.rolling(
        cfg.amount_stability_window,
        min_periods=cfg.amount_stability_window,
    ).std()
    amount_stability = amount_std.div(avg_amount.replace(0.0, np.nan))
    yield "liquidity_stability", -amount_stability
    del amount_std, amount_stability, avg_amount

    yield "short_reversal", -close.pct_change(cfg.reversal_window, fill_method=None)


def build_v5_raw_factors(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    config: V5FactorConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build interpretable V5 alpha factors from backward-looking market data only."""
    return dict(iter_v5_raw_factors(close, amount, benchmark_close, config))


def build_v5_ranked_factors(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    config: V5FactorConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or V5FactorConfig()
    return {
        name: normalize_factor(
            frame,
            lower=cfg.winsor_lower,
            upper=cfg.winsor_upper,
        )
        for name, frame in iter_v5_raw_factors(close, amount, benchmark_close, cfg)
    }


def combine_ranked_factors(
    factors: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Combine ranked factors while preserving missing-data fail-closed behavior.

    Missing factor observations are not silently imputed to zero. The score is divided
    by the absolute weight of factors that are actually present for that symbol/date.
    """
    if not weights:
        raise ValueError("at least one factor weight is required")
    unknown = sorted(set(weights) - set(factors))
    if unknown:
        raise KeyError(f"unknown factor weights: {', '.join(unknown)}")

    weighted_sum: pd.DataFrame | None = None
    available_weight: pd.DataFrame | None = None
    for name, raw_weight in weights.items():
        weight = float(raw_weight)
        if weight == 0.0:
            continue
        frame = factors[name]
        contribution = frame * weight
        present = frame.notna().astype(float) * abs(weight)
        weighted_sum = contribution if weighted_sum is None else weighted_sum.add(contribution, fill_value=0.0)
        available_weight = present if available_weight is None else available_weight.add(present, fill_value=0.0)

    if weighted_sum is None or available_weight is None:
        raise ValueError("at least one non-zero factor weight is required")
    score = weighted_sum.div(available_weight.replace(0.0, np.nan))
    return score.where(available_weight > 0.0)
