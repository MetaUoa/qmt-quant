from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest_execution import affordable_buy_quantity, deterministic_fill
from .config import CostConfig


@dataclass(frozen=True)
class BuyExecutionDecision:
    """Pre-settlement BUY outcome after fill and cash-affordability checks."""

    ready: bool
    quantity: int = 0
    execution_price: float | None = None
    blocked_random_fill: int = 0


def evaluate_buy_execution(
    *,
    requested_shares: int,
    open_price: float,
    cash: float,
    execution_date: pd.Timestamp,
    code: str,
    cost: CostConfig,
) -> BuyExecutionDecision:
    """Preserve the historical BUY short-circuit before settlement and mutation.

    The engine first resolves deterministic fill, then applies BUY slippage, then
    scales the requested order down by board lots until notional plus commission
    fits available cash. An unaffordable order remains an uncounted no-fill exactly
    as before; only deterministic-fill rejection increments the blocked counter.
    """
    if not deterministic_fill(cost, execution_date, code, "BUY"):
        return BuyExecutionDecision(ready=False, blocked_random_fill=1)

    execution_price = float(open_price) * (1.0 + float(cost.slippage_bps) / 10_000.0)
    quantity = affordable_buy_quantity(
        requested_shares=requested_shares,
        execution_price=execution_price,
        cash=cash,
        cost=cost,
    )
    if quantity <= 0:
        return BuyExecutionDecision(
            ready=False,
            quantity=0,
            execution_price=float(execution_price),
        )
    return BuyExecutionDecision(
        ready=True,
        quantity=int(quantity),
        execution_price=float(execution_price),
    )
