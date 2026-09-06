from __future__ import annotations

import pandas as pd

from qmt_quant.backtest_selection import select_rebalance_candidates
from qmt_quant.reference_data import ReferenceData


def _reference(signal_date: pd.Timestamp) -> ReferenceData:
    calendar = pd.bdate_range("2024-12-20", signal_date)
    basic = pd.DataFrame(
        {
            "ts_code": ["AAA.SZ", "BBB.SZ", "CCC.SZ", "NEW.SZ"],
            "list_date": ["20200101", "20200101", "20200101", signal_date.strftime("%Y%m%d")],
            "delist_date": [None, None, None, None],
            "exchange": ["SZSE", "SZSE", "SZSE", "SZSE"],
        }
    )
    st = pd.DataFrame(
        {
            "trade_date": [signal_date.strftime("%Y%m%d")],
            "ts_code": ["BBB.SZ"],
        }
    )
    return ReferenceData(basic, calendar, st=st)


def test_candidate_selection_preserves_score_membership_st_and_rank_order() -> None:
    signal_date = pd.Timestamp("2025-01-03")
    score_row = pd.Series(
        {
            "AAA.SZ": 0.50,
            "BBB.SZ": 0.90,
            "CCC.SZ": 0.70,
            "NEW.SZ": 1.00,
            "NAN.SZ": float("nan"),
        }
    )

    result = select_rebalance_candidates(
        score_row=score_row,
        signal_date=signal_date,
        risk_on=True,
        top_n=2,
        min_listing_sessions=2,
        reference=_reference(signal_date),
    )

    assert result.selected == ("CCC.SZ", "AAA.SZ")
    assert result.blocked_st_candidates == 1


def test_risk_off_still_preserves_historical_st_block_count() -> None:
    signal_date = pd.Timestamp("2025-01-03")
    score_row = pd.Series({"AAA.SZ": 0.5, "BBB.SZ": 0.9})

    result = select_rebalance_candidates(
        score_row=score_row,
        signal_date=signal_date,
        risk_on=False,
        top_n=1,
        min_listing_sessions=2,
        reference=_reference(signal_date),
    )

    assert result.selected == ()
    assert result.blocked_st_candidates == 1


def test_candidate_selection_without_reference_keeps_existing_top_n_behavior() -> None:
    score_row = pd.Series({"AAA.SZ": 0.2, "BBB.SZ": 0.7, "CCC.SZ": 0.4})

    result = select_rebalance_candidates(
        score_row=score_row,
        signal_date=pd.Timestamp("2025-01-03"),
        risk_on=True,
        top_n=2,
        min_listing_sessions=120,
        reference=None,
    )

    assert result.selected == ("BBB.SZ", "CCC.SZ")
    assert result.blocked_st_candidates == 0
