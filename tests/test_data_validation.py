from __future__ import annotations

import pandas as pd
import pytest

from qmt_quant.data_validation import (
    assess_akshare_crosscheck,
    assess_free_data_manifest,
    build_baseline_summary,
)


def test_manifest_validation_accepts_threshold_coverage():
    result = assess_free_data_manifest(
        {
            "source": "baostock",
            "symbols": 100,
            "adjusted_symbols_cached": 99,
            "raw_symbols_cached": 98,
            "strict_ready": False,
            "errors": [{"kind": "raw", "code": "x"}],
        },
        min_symbol_coverage=0.98,
    )
    assert result["passed"] is True
    assert result["adjusted_symbol_coverage_ratio"] == pytest.approx(0.99)
    assert result["raw_symbol_coverage_ratio"] == pytest.approx(0.98)
    assert result["download_error_kinds"] == {"raw": 1}


def test_manifest_validation_fails_when_raw_coverage_is_low():
    result = assess_free_data_manifest(
        {
            "source": "baostock",
            "symbols": 100,
            "adjusted_symbols_cached": 100,
            "raw_symbols_cached": 95,
        },
        min_symbol_coverage=0.98,
    )
    assert result["passed"] is False
    assert any("raw cache coverage" in item for item in result["failures"])


def test_akshare_crosscheck_distinguishes_mismatch_and_unavailable():
    report = pd.DataFrame(
        {
            "status": ["pass", "pass", "pass", "pass", "mismatch", "akshare_error"]
        }
    )
    result = assess_akshare_crosscheck(report, min_pass_ratio=0.8, min_compared=5)
    assert result["ready"] is True
    assert result["compared"] == 5
    assert result["pass_ratio"] == pytest.approx(0.8)
    assert result["unavailable"] == 1


def test_baseline_summary_reports_150x_and_year_stats():
    metrics = {
        "multiple": 151.0,
        "cagr": 0.88,
        "max_drawdown": -0.31,
        "sharpe": 1.7,
        "calmar": 2.8,
        "trade_count": 123,
        "symbol_coverage_ratio": 0.99,
    }
    yearly = pd.DataFrame({"year": [2023, 2024, 2025], "return": [0.2, -0.1, 0.4]})
    result = build_baseline_summary(metrics, yearly)
    assert result["target_150x_reached"] is True
    assert result["positive_years"] == 2
    assert result["negative_years"] == 1
    assert result["worst_year_return"] == pytest.approx(-0.1)
