from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from qmt_quant.research_contracts import (
    STRICT_MISSING_REFERENCE_KEYS,
    assert_strict_research_metrics,
    research_signal_eligibility,
    stitch_fold_equity,
    strict_signal_eligibility,
)


class _Reference:
    def __init__(self, dates):
        self.st_dates = set(pd.DatetimeIndex(dates))

    def filter_members(self, universe, _ts, min_listing_sessions=0):
        return list(universe)

    def st_codes(self, _ts):
        return set()


def test_signal_eligibility_requires_explicit_non_suspended_positive_same_day_amount():
    dates = pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    columns = ["000001.SZ", "000002.SZ", "000003.SZ"]
    raw_close = pd.DataFrame(10.0, index=dates, columns=columns)
    amount = pd.DataFrame(
        [[30_000_000.0, 30_000_000.0, 30_000_000.0], [30_000_000.0, 0.0, 30_000_000.0]],
        index=dates,
        columns=columns,
    )
    suspend = pd.DataFrame(
        [[0.0, 0.0, 0.0], [0.0, 0.0, float("nan")]],
        index=dates,
        columns=columns,
    )
    mask = strict_signal_eligibility(
        raw_close=raw_close,
        amount=amount,
        suspend=suspend,
        dates=dates,
        reference=_Reference(dates),
        universe=columns,
        min_price=3.0,
        min_amount=1.0,
        min_listing_sessions=0,
        amount_window=1,
        context="unit",
    )
    assert bool(mask.loc[dates[1], "000001.SZ"]) is True
    assert bool(mask.loc[dates[1], "000002.SZ"]) is False
    assert bool(mask.loc[dates[1], "000003.SZ"]) is False


def test_legacy_equivalent_profile_keeps_trailing_liquidity_semantics_explicit():
    dates = pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    code = "000001.SZ"
    raw_close = pd.DataFrame({code: [10.0, 10.0]}, index=dates)
    amount = pd.DataFrame({code: [40_000_000.0, 0.0]}, index=dates)
    legacy = research_signal_eligibility(
        raw_close=raw_close,
        amount=amount,
        dates=pd.DatetimeIndex([dates[1]]),
        reference=_Reference([dates[1]]),
        universe=[code],
        min_price=3.0,
        min_amount=10_000_000.0,
        min_listing_sessions=0,
        amount_window=2,
        require_same_day_tradable=False,
        context="legacy unit",
    )
    strict = strict_signal_eligibility(
        raw_close=raw_close,
        amount=amount,
        suspend=pd.DataFrame({code: [0.0, 0.0]}, index=dates),
        dates=pd.DatetimeIndex([dates[1]]),
        reference=_Reference([dates[1]]),
        universe=[code],
        min_price=3.0,
        min_amount=10_000_000.0,
        min_listing_sessions=0,
        amount_window=2,
        context="strict unit",
    )
    assert bool(legacy.loc[dates[1], code]) is True
    assert bool(strict.loc[dates[1], code]) is False


def test_same_day_tradability_requires_suspension_panel():
    date = pd.DatetimeIndex([pd.Timestamp("2025-01-02")])
    data = pd.DataFrame([[10.0]], index=date, columns=["000001.SZ"])
    amount = pd.DataFrame([[30_000_000.0]], index=date, columns=["000001.SZ"])
    with pytest.raises(ValueError, match="suspension panel"):
        research_signal_eligibility(
            raw_close=data,
            amount=amount,
            dates=date,
            reference=_Reference(date),
            universe=["000001.SZ"],
            min_price=3.0,
            min_amount=1.0,
            min_listing_sessions=0,
            amount_window=1,
            require_same_day_tradable=True,
        )


def test_signal_eligibility_requires_st_snapshot():
    date = pd.DatetimeIndex([pd.Timestamp("2025-01-02")])
    data = pd.DataFrame([[10.0]], index=date, columns=["000001.SZ"])
    amount = pd.DataFrame([[30_000_000.0]], index=date, columns=["000001.SZ"])
    suspend = pd.DataFrame([[0.0]], index=date, columns=["000001.SZ"])
    with pytest.raises(RuntimeError, match="missing ST snapshot"):
        strict_signal_eligibility(
            raw_close=data,
            amount=amount,
            suspend=suspend,
            dates=date,
            reference=_Reference([]),
            universe=["000001.SZ"],
            min_price=3.0,
            min_amount=1.0,
            min_listing_sessions=0,
            amount_window=1,
        )


def test_strict_metrics_include_missing_suspension_reference():
    assert "missing_suspend_rows" in STRICT_MISSING_REFERENCE_KEYS
    clean = {key: 0 for key in STRICT_MISSING_REFERENCE_KEYS}
    assert_strict_research_metrics(clean, "unit")
    dirty = dict(clean)
    dirty["missing_suspend_rows"] = 1
    with pytest.raises(RuntimeError, match="missing_suspend_rows=1"):
        assert_strict_research_metrics(dirty, "unit")


def test_fold_equity_stitching_chains_without_duplicate_boundary():
    a = pd.Series([1.0, 1.1], index=pd.to_datetime(["2021-01-01", "2021-12-31"]))
    b = pd.Series([5.0, 6.0], index=pd.to_datetime(["2022-01-01", "2022-12-31"]))
    out = stitch_fold_equity([a, b])
    assert list(out.index) == list(pd.to_datetime(["2021-01-01", "2021-12-31", "2022-12-31"]))
    assert out.iloc[-1] == pytest.approx(1.32)
