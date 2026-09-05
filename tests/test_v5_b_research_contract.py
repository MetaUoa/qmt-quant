from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_v5_b_research import (
    _assert_strict_metrics,
    _mean_row_correlation,
    _reference_availability,
)


def test_b_research_workflow_reuses_frozen_artifacts_and_never_redownloads_baostock():
    text = Path(".github/workflows/v5-b-research.yml").read_text(encoding="utf-8")
    assert 'SHARD_COUNT: "20"' in text
    assert 'SOURCE_RUN_ID: "33811845110"' in text
    assert 'RECOVERY_RUN_ID: "33887254974"' in text
    assert 'FACTOR_RUN_ID: "33954426511"' in text
    assert 'OOS_RUN_ID: "33959592406"' in text
    assert 'QMT_QUANT_CACHE_ONLY: "1"' in text
    assert "Require exactly one complete set of 20 shard artifacts" in text
    assert "Remove stale shard 13 from source run" in text
    assert "Download recovered shard 13" in text
    assert "--min-symbol-coverage 0.98" in text
    assert "--min-session-coverage 0.97" in text
    assert "prepare_free_data_shard.py" not in text
    assert "prepare_free_data.py" not in text
    assert "run_v5_b_research.py" in text


def test_reference_availability_does_not_call_liquidity_market_cap():
    basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "industry": ["bank"],
            "amount": [100_000_000.0],
        }
    )
    result = _reference_availability(basic)
    assert result["industry_neutralization_available"] is True
    assert result["market_cap_neutralization_available"] is False
    assert result["liquidity_proxy_neutralization_available"] is True
    assert "amount" not in result["market_cap_fields"]


def test_strict_metric_guard_rejects_any_missing_reference():
    _assert_strict_metrics(
        {"missing_limit_rows": 0, "missing_st_dates": 0, "missing_limit_dates": 0},
        "clean",
    )
    with pytest.raises(RuntimeError, match="missing_limit_rows=1"):
        _assert_strict_metrics(
            {"missing_limit_rows": 1, "missing_st_dates": 0, "missing_limit_dates": 0},
            "bad",
        )


def test_row_correlation_distinguishes_duplicate_and_residual_factors():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    columns = [f"s{i:03d}" for i in range(30)]
    base = pd.DataFrame(
        np.tile(np.arange(30, dtype=float), (2, 1)),
        index=dates,
        columns=columns,
    )
    duplicate = base * 2.0 + 7.0
    residual = base.copy()
    residual.iloc[0] = residual.iloc[0].to_numpy()[::-1]
    duplicate_corr = _mean_row_correlation(base, duplicate)
    residual_corr = _mean_row_correlation(base, residual)
    assert duplicate_corr["mean_cross_sectional_rank_correlation"] == pytest.approx(1.0)
    assert residual_corr["mean_cross_sectional_rank_correlation"] < 0.1
