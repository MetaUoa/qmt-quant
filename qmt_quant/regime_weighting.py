from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .factors import combine_ranked_factors


@dataclass(frozen=True)
class RegimeModel:
    train_start: str
    train_end: str
    trend_window: int
    vol_window: int
    vol_threshold: float


@dataclass(frozen=True)
class RegimeWeights:
    train_start: str
    train_end: str
    horizon: int
    global_weights: dict[str, float]
    weights_by_regime: dict[str, dict[str, float]]
    dates_by_regime: dict[str, int]


def _market_features(
    benchmark_close: pd.Series,
    *,
    trend_window: int,
    vol_window: int,
) -> pd.DataFrame:
    close = benchmark_close.sort_index().astype(float)
    ret = close.pct_change(fill_method=None)
    trend = close.div(close.shift(int(trend_window))).sub(1.0)
    vol = ret.rolling(int(vol_window), min_periods=int(vol_window)).std() * np.sqrt(252.0)
    return pd.DataFrame({"trend": trend, "vol": vol})


def fit_regime_model(
    benchmark_close: pd.Series,
    *,
    train_start,
    train_end,
    trend_window: int = 60,
    vol_window: int = 20,
    vol_quantile: float = 0.67,
    min_dates: int = 60,
) -> RegimeModel:
    """Fit only the volatility threshold on training dates; trend threshold is zero."""
    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    if end < start:
        raise ValueError("train_end must not be before train_start")
    if not 0.0 < float(vol_quantile) < 1.0:
        raise ValueError("vol_quantile must be between 0 and 1")
    features = _market_features(
        benchmark_close,
        trend_window=int(trend_window),
        vol_window=int(vol_window),
    )
    train = features.loc[(features.index >= start) & (features.index <= end)].dropna()
    if len(train) < int(min_dates):
        raise RuntimeError(f"only {len(train)} regime training dates; require {min_dates}")
    threshold = float(train["vol"].quantile(float(vol_quantile)))
    if not np.isfinite(threshold):
        raise RuntimeError("training volatility threshold is not finite")
    return RegimeModel(
        train_start=str(start.date()),
        train_end=str(end.date()),
        trend_window=int(trend_window),
        vol_window=int(vol_window),
        vol_threshold=threshold,
    )


def classify_regimes(
    benchmark_close: pd.Series,
    model: RegimeModel,
) -> pd.Series:
    features = _market_features(
        benchmark_close,
        trend_window=model.trend_window,
        vol_window=model.vol_window,
    )
    trend_state = pd.Series(
        np.where(features["trend"] >= 0.0, "up", "down"),
        index=features.index,
        dtype="object",
    )
    vol_state = pd.Series(
        np.where(features["vol"] <= float(model.vol_threshold), "calm", "volatile"),
        index=features.index,
        dtype="object",
    )
    regime = trend_state + "_" + vol_state
    return regime.where(features.notna().all(axis=1))


def _normalize_signed_weights(values: pd.Series, cap: float) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    numeric = numeric[numeric.abs() > 0.0]
    if numeric.empty:
        return {}
    capped = numeric.clip(lower=-abs(float(cap)), upper=abs(float(cap)))
    denominator = float(capped.abs().sum())
    if denominator <= 0.0:
        return {}
    return {str(name): float(value / denominator) for name, value in capped.items()}


def fit_regime_factor_weights(
    observations: pd.DataFrame,
    regimes: pd.Series,
    *,
    train_start,
    train_end,
    factors: tuple[str, ...] | list[str],
    horizon: int = 20,
    min_regime_dates: int = 12,
    weight_metric_cap: float = 0.10,
) -> RegimeWeights:
    """Fit signed IC weights by market regime using training observations only."""
    required = {"factor", "horizon", "date", "rank_ic"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"observations missing columns: {', '.join(missing)}")
    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    allowed = {str(name) for name in factors}
    frame = frame.loc[
        frame["factor"].astype(str).isin(allowed)
        & pd.to_numeric(frame["horizon"], errors="coerce").eq(int(horizon))
        & frame["date"].between(start, end)
    ].copy()
    if frame.empty:
        raise RuntimeError("no training factor observations for regime weighting")
    regime_map = regimes.copy()
    regime_map.index = pd.DatetimeIndex(regime_map.index).normalize()
    frame["regime"] = frame["date"].map(regime_map)
    frame = frame.dropna(subset=["regime", "rank_ic"])
    if frame.empty:
        raise RuntimeError("no training observations have a valid market regime")

    global_ic = frame.groupby("factor")["rank_ic"].mean()
    global_weights = _normalize_signed_weights(global_ic, weight_metric_cap)
    if not global_weights:
        raise RuntimeError("global training IC weights are empty")

    weights_by_regime: dict[str, dict[str, float]] = {}
    dates_by_regime: dict[str, int] = {}
    for regime, group in frame.groupby("regime", sort=True):
        date_count = int(group["date"].nunique())
        dates_by_regime[str(regime)] = date_count
        if date_count < int(min_regime_dates):
            weights_by_regime[str(regime)] = dict(global_weights)
            continue
        local_ic = group.groupby("factor")["rank_ic"].mean()
        local_weights = _normalize_signed_weights(local_ic, weight_metric_cap)
        weights_by_regime[str(regime)] = local_weights or dict(global_weights)

    return RegimeWeights(
        train_start=str(start.date()),
        train_end=str(end.date()),
        horizon=int(horizon),
        global_weights=dict(global_weights),
        weights_by_regime=weights_by_regime,
        dates_by_regime=dates_by_regime,
    )


def apply_regime_composite(
    ranked_factors: Mapping[str, pd.DataFrame],
    regimes: pd.Series,
    fitted: RegimeWeights,
) -> pd.DataFrame:
    """Apply frozen regime weights; unseen regimes use frozen global training weights."""
    if not ranked_factors:
        raise ValueError("ranked_factors must not be empty")
    first = next(iter(ranked_factors.values()))
    output = pd.DataFrame(np.nan, index=first.index, columns=first.columns, dtype=float)
    regime_series = regimes.reindex(first.index)
    for regime in regime_series.dropna().astype(str).unique():
        dates = regime_series.index[regime_series.astype("string").eq(regime)]
        weights = fitted.weights_by_regime.get(regime, fitted.global_weights)
        subset = {name: panel.reindex(dates) for name, panel in ranked_factors.items() if name in weights}
        if not subset:
            raise RuntimeError(f"no factor panel available for regime {regime}")
        output.loc[dates] = combine_ranked_factors(subset, weights).reindex(dates)
    return output
