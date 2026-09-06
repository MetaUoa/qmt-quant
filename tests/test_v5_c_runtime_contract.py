from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from qmt_quant.research_runtime import install_v5_c_contracts
from qmt_quant.v5_gates import evaluate_basic_alpha_gate


class _Reference:
    def __init__(self, dates):
        self.st_dates = set(pd.DatetimeIndex(dates))

    def filter_members(self, universe, _ts, min_listing_sessions=0):
        return list(universe)

    def st_codes(self, _ts):
        return set()


def test_c_runtime_uses_same_day_suspension_and_turnover_contract():
    module = SimpleNamespace()
    install_v5_c_contracts(module)
    dates = pd.DatetimeIndex(["2025-01-02"])
    columns = ["000001.SZ", "000002.SZ"]
    raw_close = pd.DataFrame([[10.0, 10.0]], index=dates, columns=columns)
    amount = pd.DataFrame([[30_000_000.0, 0.0]], index=dates, columns=columns)
    suspend = pd.DataFrame([[0.0, 0.0]], index=dates, columns=columns)
    mask = module._eligible_mask(
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
    )
    assert bool(mask.loc[dates[0], "000001.SZ"]) is True
    assert bool(mask.loc[dates[0], "000002.SZ"]) is False


def test_c_runtime_strict_metrics_include_missing_suspension():
    module = SimpleNamespace()
    install_v5_c_contracts(module)
    with pytest.raises(RuntimeError, match="missing_suspend_rows=1"):
        module._assert_strict_metrics(
            {
                "missing_limit_rows": 0,
                "missing_st_dates": 0,
                "missing_limit_dates": 0,
                "missing_suspend_rows": 1,
            },
            "C outer",
        )


def test_c_runtime_uses_central_basic_alpha_gate():
    module = SimpleNamespace()
    install_v5_c_contracts(module)
    assert module._basic_alpha_gate is evaluate_basic_alpha_gate


def test_c7_entrypoint_installs_canonical_contracts_without_touching_holdout():
    text = open("run_v5_c7_nested_research.py", encoding="utf-8").read()
    assert "install_v5_c_contracts(c1)" in text
    assert "202601" not in text
