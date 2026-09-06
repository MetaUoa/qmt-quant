from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import run_v5_b_canonical_research as b_entry
import run_v5_composite_canonical_oos as composite_entry
from qmt_quant.research_runtime import install_legacy_v5_research_contracts


class _Reference:
    def __init__(self, dates):
        self.st_dates = set(pd.DatetimeIndex(dates))

    def filter_members(self, universe, _ts, min_listing_sessions=0):
        return list(universe)

    def st_codes(self, _ts):
        return set()


def _module_surface():
    return SimpleNamespace(
        _coverage_or_fail=lambda *args, **kwargs: None,
        _eligible_mask=lambda *args, **kwargs: None,
        _assert_strict_metrics=lambda *args, **kwargs: None,
        _stitch_fold_equity=lambda *args, **kwargs: None,
    )


def test_legacy_adapter_preserves_trailing_liquidity_profile():
    module = _module_surface()
    install_legacy_v5_research_contracts(module, context="adapter unit")
    dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03"])
    code = "000001.SZ"
    raw_close = pd.DataFrame({code: [10.0, 10.0]}, index=dates)
    amount = pd.DataFrame({code: [40_000_000.0, 0.0]}, index=dates)
    mask = module._eligible_mask(
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


def test_b_adapter_upgrades_strict_metric_guard_to_missing_suspension():
    module = _module_surface()
    install_legacy_v5_research_contracts(module, context="B research")
    with pytest.raises(RuntimeError, match="missing_suspend_rows=1"):
        module._assert_strict_metrics(
            {
                "missing_limit_rows": 0,
                "missing_st_dates": 0,
                "missing_limit_dates": 0,
                "missing_suspend_rows": 1,
            },
            "B fold",
        )


def test_canonical_entrypoints_refuse_2026_before_runtime_switch():
    b_entry._assert_pre_2026_only(["--end", "2025-12-31"])
    composite_entry._assert_pre_2026_only(["--end", "20251231"])
    with pytest.raises(RuntimeError, match="holdout remains blinded"):
        b_entry._assert_pre_2026_only(["--end", "20260101"])
    with pytest.raises(RuntimeError, match="holdout remains blinded"):
        composite_entry._assert_pre_2026_only(["--end", "2026-01-01"])
