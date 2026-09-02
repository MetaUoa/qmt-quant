from __future__ import annotations

import json

import pandas as pd
import pytest

from validate_backtest_output import validate_output


def _write_valid_output(root, *, coverage_ratio=1.0, missing_limit_rows=0):
    root.mkdir(parents=True, exist_ok=True)
    metrics = {
        "point_in_time_universe": True,
        "strict_reference": True,
        "raw_limit_reference": True,
        "missing_st_dates": 0,
        "missing_limit_dates": 0,
        "missing_limit_rows": missing_limit_rows,
        "symbol_coverage_ratio": coverage_ratio,
        "multiple": 1.2,
        "max_drawdown": -0.1,
        "sharpe": 1.0,
    }
    quality = {"symbol_coverage_ratio": coverage_ratio, "raw_limit_reference_coverage_ratio": coverage_ratio}
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (root / "data_quality.json").write_text(json.dumps(quality), encoding="utf-8")
    pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "equity": [1_000_000.0, 1_200_000.0]}).to_csv(
        root / "equity.csv", index=False
    )
    pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "signal_date": ["2020-01-01"],
            "code": ["AAA.SZ"],
            "side": ["BUY"],
            "shares": [100],
        }
    ).to_csv(root / "trades.csv", index=False)
    pd.DataFrame({"code": ["AAA.SZ"], "loaded": [True]}).to_csv(root / "universe_coverage.csv", index=False)


def test_validate_output_accepts_strict_complete_result(tmp_path):
    _write_valid_output(tmp_path)
    result = validate_output(tmp_path, 0.95)
    assert result["passed"] is True
    assert result["trade_count"] == 1


def test_validate_output_rejects_missing_limit_rows(tmp_path):
    _write_valid_output(tmp_path, missing_limit_rows=1)
    with pytest.raises(AssertionError, match="per-symbol price-limit rows are incomplete"):
        validate_output(tmp_path, 0.95)


def test_validate_output_rejects_low_symbol_coverage(tmp_path):
    _write_valid_output(tmp_path, coverage_ratio=0.90)
    with pytest.raises(AssertionError, match="symbol coverage"):
        validate_output(tmp_path, 0.95)
