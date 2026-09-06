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
from qmt_quant.workflow_contract import env_value, load_workflow, normalized_run, step, structured_text


def test_b_research_workflow_reuses_frozen_artifacts_and_never_redownloads_baostock():
    workflow = load_workflow(Path(".github/workflows/v5-b-research.yml"))
    assert workflow.get("on") == {"workflow_dispatch": None}
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "SOURCE_RUN_ID") == "33811845110"
    assert env_value(workflow, "RECOVERY_RUN_ID") == "33887254974"
    assert env_value(workflow, "FACTOR_RUN_ID") == "33954426511"
    assert env_value(workflow, "OOS_RUN_ID") == "33959592406"
    assert env_value(workflow, "DATA_END") == "20251231"
    assert env_value(workflow, "QMT_QUANT_CACHE_ONLY") == "1"
    for name in (
        "Require exactly one complete set of 20 shard artifacts",
        "Remove stale shard 13 from source run",
        "Download recovered shard 13",
    ):
        assert step(workflow, "v5-b-research", name)
    audit = normalized_run(
        workflow, "v5-b-research", "Revalidate full historical data before B research"
    )
    runner = normalized_run(workflow, "v5-b-research", "Run strict B1-B6 research")
    assert "--min-symbol-coverage 0.98" in audit
    assert "--min-session-coverage 0.97" in audit
    assert "run_v5_b_canonical_research.py" in runner
    assert "run_v5_b_research.py" not in runner
    semantic = structured_text(workflow)
    assert "prepare_free_data_shard.py" not in semantic
    assert "prepare_free_data.py" not in semantic


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
