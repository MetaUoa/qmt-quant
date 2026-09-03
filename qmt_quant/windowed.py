from __future__ import annotations

from typing import Dict

import pandas as pd

from .backtest import BacktestResult, calculate_metrics, run_backtest
from .config import CostConfig, StrategyConfig
from .reference_data import ReferenceData


def _slice_frames(
    frames: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for code, frame in frames.items():
        if frame is None or frame.empty:
            continue
        sliced = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
        if not sliced.empty:
            out[code] = sliced
    return out


def context_start_for_window(
    bars: Dict[str, pd.DataFrame],
    benchmark_code: str,
    strategy: StrategyConfig,
    trade_start,
) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    if benchmark_code not in bars or bars[benchmark_code].empty:
        raise ValueError(f"Benchmark {benchmark_code} is missing from bars.")
    calendar = pd.DatetimeIndex(bars[benchmark_code].index).sort_values().unique()
    requested = pd.Timestamp(trade_start).normalize()
    trade_i = int(calendar.searchsorted(requested, side="left"))
    if trade_i >= len(calendar):
        raise ValueError(f"trade_start {requested.date()} is after available benchmark data.")
    actual_start = pd.Timestamp(calendar[trade_i])
    pre_sessions = max(
        int(strategy.warmup) + max(int(strategy.execution_delay_sessions), 1) - 1,
        0,
    )
    context_i = max(trade_i - pre_sessions, 0)
    truncated = bool(trade_i < pre_sessions)
    return pd.Timestamp(calendar[context_i]), actual_start, truncated


def run_window_backtest(
    bars: Dict[str, pd.DataFrame],
    benchmark_code: str,
    strategy: StrategyConfig | None = None,
    costs: CostConfig | None = None,
    *,
    trade_start,
    trade_end,
    reference: ReferenceData | None = None,
    strict_reference: bool = False,
    limit_reference_bars: Dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    cfg = strategy or StrategyConfig()
    cost = costs or CostConfig()
    context_start, actual_start, warmup_truncated = context_start_for_window(
        bars, benchmark_code, cfg, trade_start
    )
    end = pd.Timestamp(trade_end).normalize()
    if end < actual_start:
        raise ValueError("trade_end must not be before trade_start.")

    window_bars = _slice_frames(bars, context_start, end)
    window_raw = (
        _slice_frames(limit_reference_bars, context_start, end)
        if limit_reference_bars is not None
        else None
    )
    raw = run_backtest(
        window_bars,
        benchmark_code,
        cfg,
        cost,
        reference=reference,
        strict_reference=strict_reference,
        limit_reference_bars=window_raw,
    )

    equity = raw.equity.loc[
        (raw.equity.index >= actual_start) & (raw.equity.index <= end)
    ].copy()
    trades = raw.trades.copy()
    if not trades.empty and "date" in trades:
        dates = pd.to_datetime(trades["date"], errors="coerce")
        trades = trades.loc[(dates >= actual_start) & (dates <= end)].copy()

    metrics = calculate_metrics(equity["equity"]) if not equity.empty else {}
    return_keys = set(metrics)
    for key, value in raw.metrics.items():
        if key not in return_keys:
            metrics[key] = value
    metrics.update(
        {
            "window_trade_start": str(actual_start.date()),
            "window_trade_end": str(end.date()),
            "window_context_start": str(context_start.date()),
            "warmup_truncated": warmup_truncated,
        }
    )
    config = dict(raw.config)
    config["window"] = {
        "trade_start": str(actual_start.date()),
        "trade_end": str(end.date()),
        "context_start": str(context_start.date()),
        "warmup_truncated": warmup_truncated,
    }
    return BacktestResult(equity=equity, trades=trades, metrics=metrics, config=config)
