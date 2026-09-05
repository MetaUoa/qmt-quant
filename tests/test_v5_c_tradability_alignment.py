import pandas as pd

from run_v5_c_nested_research import _eligible_mask


class _ReferenceStub:
    def __init__(self, dates, members):
        self.st_dates = frozenset(pd.DatetimeIndex(dates))
        self._members = list(members)

    def filter_members(self, codes, date, min_listing_sessions=0):
        del date, min_listing_sessions
        return [code for code in codes if code in self._members]

    def st_codes(self, date):
        del date
        return set()


def test_c_eligible_mask_excludes_same_day_suspended_zero_amount_and_missing_suspend_flag():
    dates = pd.DatetimeIndex(["2025-01-02"])
    codes = ["A.SZ", "B.SZ", "C.SZ", "D.SZ"]
    raw_close = pd.DataFrame([[10.0, 10.0, 10.0, 10.0]], index=dates, columns=codes)
    amount = pd.DataFrame([[100.0, 100.0, 0.0, 100.0]], index=dates, columns=codes)
    suspend = pd.DataFrame([[0.0, 1.0, 0.0, float("nan")]], index=dates, columns=codes)

    mask = _eligible_mask(
        raw_close=raw_close,
        amount=amount,
        suspend=suspend,
        dates=dates,
        reference=_ReferenceStub(dates, codes),
        universe=codes,
        min_price=3.0,
        min_amount=1.0,
        min_listing_sessions=120,
        amount_window=1,
    )

    assert bool(mask.loc[dates[0], "A.SZ"])
    assert not bool(mask.loc[dates[0], "B.SZ"])
    assert not bool(mask.loc[dates[0], "C.SZ"])
    assert not bool(mask.loc[dates[0], "D.SZ"])


def test_c_tradability_alignment_does_not_lower_exposure_threshold_contract():
    text = open(".github/workflows/v5-c-nested-research.yml", encoding="utf-8").read()
    assert "--min-exposure-coverage 0.95" in text
    assert "--min-symbol-coverage 0.98" in text
    assert "--min-session-coverage 0.97" in text
