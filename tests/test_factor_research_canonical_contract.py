from __future__ import annotations

import pandas as pd

from run_v5_factor_research import _factor_research_eligibility


class _Reference:
    def __init__(self, dates):
        self.st_dates = set(pd.DatetimeIndex(dates))

    def filter_members(self, universe, _ts, min_listing_sessions=0):
        return list(universe)

    def st_codes(self, _ts):
        return set()


def test_factor_research_keeps_legacy_equivalent_trailing_liquidity_profile() -> None:
    dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03"])
    code = "000001.SZ"
    raw_close = pd.DataFrame({code: [10.0, 10.0]}, index=dates)
    amount = pd.DataFrame({code: [40_000_000.0, 0.0]}, index=dates)
    mask = _factor_research_eligibility(
        raw_close=raw_close,
        amount=amount,
        dates=pd.DatetimeIndex([dates[1]]),
        reference=_Reference([dates[1]]),
        universe=[code],
        min_price=3.0,
        min_amount=10_000_000.0,
        min_listing_sessions=0,
        amount_window=2,
    )
    assert bool(mask.loc[dates[1], code]) is True


def test_factor_research_still_excludes_pit_st_codes() -> None:
    class _StReference(_Reference):
        def st_codes(self, _ts):
            return {"000001.SZ"}

    date = pd.DatetimeIndex(["2025-01-02"])
    code = "000001.SZ"
    raw_close = pd.DataFrame({code: [10.0]}, index=date)
    amount = pd.DataFrame({code: [40_000_000.0]}, index=date)
    mask = _factor_research_eligibility(
        raw_close=raw_close,
        amount=amount,
        dates=date,
        reference=_StReference(date),
        universe=[code],
        min_price=3.0,
        min_amount=10_000_000.0,
        min_listing_sessions=0,
        amount_window=1,
    )
    assert bool(mask.loc[date[0], code]) is False
