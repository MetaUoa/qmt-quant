from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BacktestDiagnostics:
    """Execution diagnostics needed to assemble the public backtest metrics payload."""

    trade_count: int
    rebalance_count: int
    initial_cash: float
    blocked_st_candidates: int
    blocked_limit_buys: int
    blocked_limit_sells: int
    blocked_suspended: int
    blocked_t1_sells: int
    missing_suspend_rows: int
    missing_limit_rows: int
    missing_st_dates: int
    missing_limit_dates: int
    point_in_time_universe: bool
    strict_reference: bool
    raw_limit_reference: bool
    blocked_random_fill: int
    execution_delay_sessions: int
    fill_probability: float
    average_market_breadth: float
    score_override: bool = False
    risk_on_override: bool = False


def assemble_backtest_metrics(
    base_metrics: Mapping[str, object],
    diagnostics: BacktestDiagnostics,
) -> dict[str, object]:
    """Purely assemble the existing public metrics/report contract.

    Execution policy stays outside this helper. The fixed boundary flags are report
    invariants of the current daily-bar engine and intentionally retain their exact
    historical values and wording.
    """
    metrics: dict[str, object] = dict(base_metrics)
    metrics.update(
        {
            "trade_count": int(diagnostics.trade_count),
            "rebalance_count": int(diagnostics.rebalance_count),
            "initial_cash": float(diagnostics.initial_cash),
            "blocked_st_candidates": int(diagnostics.blocked_st_candidates),
            "blocked_limit_buys": int(diagnostics.blocked_limit_buys),
            "blocked_limit_sells": int(diagnostics.blocked_limit_sells),
            "blocked_suspended": int(diagnostics.blocked_suspended),
            "blocked_t1_sells": int(diagnostics.blocked_t1_sells),
            "missing_suspend_rows": int(diagnostics.missing_suspend_rows),
            "missing_limit_rows": int(diagnostics.missing_limit_rows),
            "missing_st_dates": int(diagnostics.missing_st_dates),
            "missing_limit_dates": int(diagnostics.missing_limit_dates),
            "point_in_time_universe": bool(diagnostics.point_in_time_universe),
            "strict_reference": bool(diagnostics.strict_reference),
            "raw_limit_reference": bool(diagnostics.raw_limit_reference),
            "blocked_random_fill": int(diagnostics.blocked_random_fill),
            "execution_delay_sessions": int(diagnostics.execution_delay_sessions),
            "fill_probability": float(diagnostics.fill_probability),
            "t_plus_one_enforced": True,
            "limit_model": "open_auction_reference_plus_one_price_daily_fallback",
            "intraday_limit_touch_modelled": False,
            "average_market_breadth": float(diagnostics.average_market_breadth),
        }
    )
    if diagnostics.score_override:
        metrics["score_override"] = True
    if diagnostics.risk_on_override:
        metrics["risk_on_override"] = True
    return metrics
