from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .backtest import _build_features, _panel
from .config import StrategyConfig
from .reference_data import ReferenceData


def latest_target_codes(
    bars: Dict[str, pd.DataFrame],
    benchmark_code: str,
    cfg: StrategyConfig,
    *,
    reference: ReferenceData | None = None,
    signal_date=None,
    raw_bars: Dict[str, pd.DataFrame] | None = None,
    strict_st: bool = True,
) -> tuple[pd.Timestamp, list[str], dict]:
    if benchmark_code not in bars:
        raise ValueError(f"Benchmark {benchmark_code} missing")
    calendar = pd.DatetimeIndex(bars[benchmark_code].index).sort_values()
    if len(calendar) < cfg.warmup:
        raise ValueError(f"Need at least {cfg.warmup} benchmark sessions, got {len(calendar)}")
    if signal_date is None:
        ts = calendar[-1]
    else:
        target = pd.Timestamp(signal_date).normalize()
        available = calendar[calendar <= target]
        if len(available) == 0:
            raise ValueError("No benchmark session on/before requested signal date")
        ts = available[-1]

    stock_bars = {k: v for k, v in bars.items() if k != benchmark_code}
    close = _panel(stock_bars, "close", calendar)
    amount = _panel(stock_bars, "amount", calendar)
    if raw_bars:
        raw_stock = {k: v for k, v in raw_bars.items() if k != benchmark_code}
        raw_close = _panel(raw_stock, "close", calendar)
    else:
        raw_close = close
    score, _ = _build_features(close, amount, cfg, eligibility_price=raw_close)

    benchmark_close = bars[benchmark_code]["close"].reindex(calendar).ffill()
    benchmark_ma = benchmark_close.rolling(cfg.benchmark_ma, min_periods=cfg.benchmark_ma).mean()
    benchmark_mom = benchmark_close.pct_change(cfg.benchmark_mom_days, fill_method=None)
    breadth_ma = close.rolling(cfg.breadth_ma, min_periods=cfg.breadth_ma).mean()
    breadth = (close > breadth_ma).sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan)
    breadth_value = float(breadth.loc[ts]) if pd.notna(breadth.loc[ts]) else 0.0
    risk_on = bool(
        benchmark_close.loc[ts] > benchmark_ma.loc[ts]
        and benchmark_mom.loc[ts] > cfg.benchmark_mom_floor
        and breadth_value >= cfg.min_breadth
    )

    row = score.loc[ts].dropna().sort_values(ascending=False)
    if reference is not None:
        if strict_st and ts not in reference.st_dates:
            raise ValueError(f"Missing ST snapshot for live signal date {ts.date()}")
        members = set(reference.filter_members(row.index, ts, cfg.min_listing_sessions))
        row = row[row.index.isin(members)]
        row = row.drop(index=row.index.intersection(reference.st_codes(ts)), errors="ignore")
    selected = list(row.head(cfg.top_n).index) if risk_on else []
    diagnostics = {
        "signal_date": str(ts.date()),
        "risk_on": risk_on,
        "market_breadth": breadth_value,
        "candidate_count": int(len(row)),
        "selected_count": int(len(selected)),
    }
    return ts, selected, diagnostics
