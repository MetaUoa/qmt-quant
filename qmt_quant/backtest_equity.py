from __future__ import annotations

import pandas as pd


def build_equity_row(
    *,
    date: pd.Timestamp,
    equity: float,
    cash: float,
    position_count: int,
    risk_on: bool,
) -> dict[str, object]:
    """Build one public daily equity-row payload without side effects."""
    return {
        "date": date,
        "equity": float(equity),
        "cash": float(cash),
        "positions": int(position_count),
        "risk_on": bool(risk_on),
    }
