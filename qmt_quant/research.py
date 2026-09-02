from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import BacktestResult, calculate_metrics
from .config import StrategyConfig
from .reporting import yearly_returns


def trade_concentration(trades: pd.DataFrame) -> float:
    """Share of total traded notional attributable to the busiest symbol."""
    if trades is None or trades.empty or "code" not in trades or "notional" not in trades:
        return 0.0
    notionals = pd.to_numeric(trades["notional"], errors="coerce").fillna(0.0).abs()
    grouped = notionals.groupby(trades["code"].astype(str)).sum()
    total = float(grouped.sum())
    return float(grouped.max() / total) if total > 0 and len(grouped) else 0.0


def window_metrics(result: BacktestResult, start, end) -> dict:
    eq = result.equity.loc[
        (result.equity.index >= pd.Timestamp(start)) & (result.equity.index <= pd.Timestamp(end)),
        "equity",
    ]
    return calculate_metrics(eq)


def research_score(
    result: BacktestResult,
    start=None,
    end=None,
    *,
    max_drawdown: float = 0.50,
    concentration_soft_limit: float = 0.35,
) -> tuple[float, dict]:
    """Multi-objective score that penalizes drawdown, annual instability and concentration.

    This is intentionally not a pure return objective. A candidate with a spectacular CAGR
    but unstable annual returns or one-symbol turnover concentration is downgraded.
    """
    if start is not None or end is not None:
        left = pd.Timestamp(start) if start is not None else result.equity.index.min()
        right = pd.Timestamp(end) if end is not None else result.equity.index.max()
        eq = result.equity.loc[(result.equity.index >= left) & (result.equity.index <= right), "equity"]
        metrics = calculate_metrics(eq)
        trades = result.trades
        if not trades.empty and "date" in trades:
            d = pd.to_datetime(trades["date"], errors="coerce")
            trades = trades.loc[(d >= left) & (d <= right)]
    else:
        eq = result.equity["equity"]
        metrics = result.metrics
        trades = result.trades

    if not metrics:
        return float("-inf"), {}
    dd = abs(float(metrics.get("max_drawdown", -1.0)))
    if dd > float(max_drawdown):
        return float("-inf"), {"reject_reason": "max_drawdown"}

    yr = yearly_returns(eq)
    annual = pd.to_numeric(yr.get("return", pd.Series(dtype=float)), errors="coerce").dropna()
    annual_std = float(annual.std(ddof=0)) if len(annual) > 1 else 0.0
    negative_years = int((annual < 0).sum()) if len(annual) else 0
    concentration = trade_concentration(trades)
    concentration_penalty = max(concentration - concentration_soft_limit, 0.0)
    instability_penalty = 0.35 * annual_std + 0.12 * negative_years + 0.80 * concentration_penalty

    score = (
        0.38 * float(metrics.get("calmar", 0.0))
        + 0.27 * float(metrics.get("sharpe", 0.0))
        + 0.25 * float(metrics.get("cagr", 0.0))
        + 0.10 * np.log1p(max(float(metrics.get("multiple", 1.0)) - 1.0, 0.0))
        - instability_penalty
    )
    diagnostics = {
        "research_score": float(score),
        "annual_return_std": annual_std,
        "negative_years": negative_years,
        "trade_concentration": concentration,
        "instability_penalty": float(instability_penalty),
    }
    return float(score), diagnostics


def make_candidate_grid(
    base: StrategyConfig | None = None,
    *,
    top_n: Iterable[int] = (5, 8, 12),
    rebalance_days: Iterable[int] = (3, 5, 10),
    min_momentum: Iterable[float] = (0.00, 0.02, 0.05),
    max_daily_vol: Iterable[float] = (0.06, 0.075, 0.09),
    min_breadth: Iterable[float] = (0.0, 0.40, 0.50),
    factor_mix: Iterable[tuple[float, float, float, float]] = (
        (0.20, 0.30, 0.50, 0.75),
        (0.15, 0.35, 0.50, 0.65),
        (0.30, 0.30, 0.40, 0.85),
    ),
) -> list[StrategyConfig]:
    base = base or StrategyConfig()
    out: list[StrategyConfig] = []
    for n, reb, mom, max_vol, breadth, mix in product(
        top_n, rebalance_days, min_momentum, max_daily_vol, min_breadth, factor_mix
    ):
        ws, wm, wl, vp = mix
        out.append(
            replace(
                base,
                top_n=int(n),
                rebalance_days=int(reb),
                min_momentum=float(mom),
                max_daily_vol=float(max_vol),
                min_breadth=float(breadth),
                weight_short=float(ws),
                weight_mid=float(wm),
                weight_long=float(wl),
                vol_penalty=float(vp),
            )
        )
    return out


def config_key(cfg: StrategyConfig) -> str:
    return (
        f"top{cfg.top_n}_reb{cfg.rebalance_days}_mom{cfg.min_momentum:.3f}"
        f"_vol{cfg.max_daily_vol:.3f}_breadth{cfg.min_breadth:.2f}"
        f"_w{cfg.weight_short:.2f}-{cfg.weight_mid:.2f}-{cfg.weight_long:.2f}"
        f"_vp{cfg.vol_penalty:.2f}"
    )


def parameter_distance(a: StrategyConfig, b: StrategyConfig) -> float:
    va = np.array(
        [
            a.top_n / 10.0,
            a.rebalance_days / 5.0,
            a.min_momentum / 0.03,
            a.max_daily_vol / 0.075,
            a.min_breadth / 0.50 if a.min_breadth or b.min_breadth else 0.0,
            a.weight_short,
            a.weight_mid,
            a.weight_long,
            a.vol_penalty,
        ],
        dtype=float,
    )
    vb = np.array(
        [
            b.top_n / 10.0,
            b.rebalance_days / 5.0,
            b.min_momentum / 0.03,
            b.max_daily_vol / 0.075,
            b.min_breadth / 0.50 if a.min_breadth or b.min_breadth else 0.0,
            b.weight_short,
            b.weight_mid,
            b.weight_long,
            b.vol_penalty,
        ],
        dtype=float,
    )
    return float(np.linalg.norm(va - vb))


def add_neighborhood_stability(rows: pd.DataFrame, configs: dict[str, StrategyConfig], neighbors: int = 4) -> pd.DataFrame:
    """Penalize isolated parameter spikes using nearby candidate scores."""
    out = rows.copy()
    if out.empty or "candidate" not in out or "raw_score" not in out:
        return out
    stable_scores: list[float] = []
    neighbor_dispersion: list[float] = []
    for row in out.itertuples(index=False):
        key = str(row.candidate)
        cfg = configs[key]
        scored = []
        for other in out.itertuples(index=False):
            other_key = str(other.candidate)
            if other_key == key:
                continue
            score = float(other.raw_score)
            if not np.isfinite(score):
                continue
            scored.append((parameter_distance(cfg, configs[other_key]), score))
        scored.sort(key=lambda x: x[0])
        vals = [x[1] for x in scored[: max(int(neighbors), 1)]]
        dispersion = float(np.std(vals)) if vals else 0.0
        median_neighbor = float(np.median(vals)) if vals else float(row.raw_score)
        spike = max(float(row.raw_score) - median_neighbor, 0.0)
        stable = float(row.raw_score) - 0.25 * dispersion - 0.20 * spike
        stable_scores.append(stable)
        neighbor_dispersion.append(dispersion)
    out["neighbor_dispersion"] = neighbor_dispersion
    out["stable_score"] = stable_scores
    return out.sort_values("stable_score", ascending=False).reset_index(drop=True)


def config_dict(cfg: StrategyConfig) -> dict:
    return asdict(cfg)
