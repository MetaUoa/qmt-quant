from __future__ import annotations

import pandas as pd

from .backtest_execution import BuySettlement, SellSettlement


def build_sell_trade_row(
    *,
    execution_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    code: str,
    shares: int,
    execution_price: float,
    settlement: SellSettlement,
) -> dict[str, object]:
    """Build the existing public SELL trade-record payload without side effects."""
    return {
        "date": execution_date,
        "code": str(code),
        "side": "SELL",
        "shares": int(shares),
        "price": float(execution_price),
        "notional": settlement.notional,
        "commission": settlement.commission,
        "stamp_tax": settlement.stamp_tax,
        "signal_date": signal_date,
    }


def build_buy_trade_row(
    *,
    execution_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    code: str,
    shares: int,
    execution_price: float,
    settlement: BuySettlement,
) -> dict[str, object]:
    """Build the existing public BUY trade-record payload without side effects."""
    return {
        "date": execution_date,
        "code": str(code),
        "side": "BUY",
        "shares": int(shares),
        "price": float(execution_price),
        "notional": settlement.notional,
        "commission": settlement.commission,
        "stamp_tax": 0.0,
        "signal_date": signal_date,
    }
