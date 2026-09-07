from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest_execution import TradabilityGuard, deterministic_fill
from .config import CostConfig


@dataclass(frozen=True)
class SellExecutionDecision:
    """Pre-settlement SELL decision preserving the engine's short-circuit order."""

    ready: bool
    blocked_t1_sells: int = 0
    blocked_suspended: int = 0
    blocked_limit_sells: int = 0
    blocked_random_fill: int = 0


def t1_sell_allowed(
    last_buy_date: pd.Timestamp | None,
    execution_date: pd.Timestamp,
) -> bool:
    """Return whether inventory predates the execution session under A-share T+1."""
    if last_buy_date is None:
        return True
    return pd.Timestamp(last_buy_date).normalize() < pd.Timestamp(execution_date).normalize()


def evaluate_sell_execution(
    *,
    last_buy_date: pd.Timestamp | None,
    execution_date: pd.Timestamp,
    code: str,
    cost: CostConfig,
    guard: TradabilityGuard,
) -> SellExecutionDecision:
    """Apply the historical SELL eligibility/fill checks without settlement or mutation.

    Ordering is intentionally fixed: T+1 first, then suspension, then SELL price-limit,
    then deterministic fill. The first failed check stops evaluation exactly as the
    original inline loop did.
    """
    if not t1_sell_allowed(last_buy_date, execution_date):
        return SellExecutionDecision(ready=False, blocked_t1_sells=1)
    if guard.is_halted(execution_date, code):
        return SellExecutionDecision(ready=False, blocked_suspended=1)
    if guard.limit_blocked(execution_date, code, "SELL"):
        return SellExecutionDecision(ready=False, blocked_limit_sells=1)
    if not deterministic_fill(cost, execution_date, code, "SELL"):
        return SellExecutionDecision(ready=False, blocked_random_fill=1)
    return SellExecutionDecision(ready=True)
