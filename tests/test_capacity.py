from __future__ import annotations

import pandas as pd

from qmt_quant.capacity import exclude_bottom_liquidity, summarize_capacity, trade_capacity_report


def test_bottom_liquidity_exclusion_is_date_local_and_missing_is_false():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    amount = pd.DataFrame(
        {
            "A": [10.0, 100.0],
            "B": [20.0, 50.0],
            "C": [30.0, None],
            "D": [40.0, 25.0],
        },
        index=idx,
    )
    mask = exclude_bottom_liquidity(amount, bottom_fraction=0.25)
    assert mask.loc[idx[0], "A"] is False or not bool(mask.loc[idx[0], "A"])
    assert bool(mask.loc[idx[0], "D"])
    assert not bool(mask.loc[idx[1], "C"])


def test_capacity_report_fails_closed_when_daily_amount_is_missing():
    date = pd.Timestamp("2020-01-02")
    trades = pd.DataFrame(
        {
            "date": [date, date],
            "code": ["A", "B"],
            "notional": [10_000.0, 10_000.0],
        }
    )
    amount = pd.DataFrame({"A": [1_000_000.0]}, index=[date])
    report = trade_capacity_report(trades, amount, max_participation=0.10)
    assert bool(report.loc[0, "capacity_pass"])
    assert not bool(report.loc[1, "capacity_reference_present"])
    assert not bool(report.loc[1, "capacity_pass"])
    summary = summarize_capacity(report, max_participation=0.10)
    assert summary["reference_coverage"] == 0.5
    assert summary["passed"] is False


def test_capacity_summary_passes_only_when_every_trade_is_within_threshold():
    date = pd.Timestamp("2020-01-02")
    trades = pd.DataFrame(
        {
            "date": [date, date],
            "code": ["A", "B"],
            "notional": [10_000.0, 20_000.0],
        }
    )
    amount = pd.DataFrame({"A": [1_000_000.0], "B": [2_000_000.0]}, index=[date])
    report = trade_capacity_report(trades, amount, max_participation=0.10)
    summary = summarize_capacity(report, max_participation=0.10)
    assert summary["reference_coverage"] == 1.0
    assert summary["capacity_pass_ratio"] == 1.0
    assert summary["passed"] is True
