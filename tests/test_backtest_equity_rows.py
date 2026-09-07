from __future__ import annotations

import pandas as pd

from qmt_quant.backtest_equity import build_equity_row


def test_build_equity_row_preserves_public_payload_types_and_values() -> None:
    ts = pd.Timestamp("2025-01-06")
    row = build_equity_row(
        date=ts,
        equity=100_123.45,
        cash=12_345.67,
        position_count=3,
        risk_on=True,
    )

    assert row == {
        "date": ts,
        "equity": 100_123.45,
        "cash": 12_345.67,
        "positions": 3,
        "risk_on": True,
    }
