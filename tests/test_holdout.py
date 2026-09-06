from pathlib import Path

import pytest

from qmt_quant.holdout import (
    FrozenCandidate,
    assert_holdout_boundary,
    freeze_candidate_manifest,
    verify_candidate_manifest,
)
from qmt_quant.workflow_contract import (
    env_value,
    load_workflow,
    matrix_values,
    max_parallel,
    normalized_run,
)


def _candidate():
    return FrozenCandidate(
        name="v5-c",
        research_data_end="2025-12-31",
        neutralization_variant="liquidity",
        weights={"low_volatility": 0.5, "liquidity_stability": 0.5},
        top_n=8,
        rebalance_days=5,
        execution_delay_sessions=1,
        min_price=3.0,
        min_amount=20_000_000.0,
        min_listing_sessions=120,
    )


def test_frozen_candidate_fingerprint_is_stable_and_tamper_evident():
    candidate = _candidate()
    manifest = freeze_candidate_manifest(candidate)
    assert verify_candidate_manifest(manifest).fingerprint() == candidate.fingerprint()
    manifest["candidate"]["top_n"] = 9
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        verify_candidate_manifest(manifest)


def test_holdout_must_start_after_research_data_end():
    candidate = _candidate()
    assert_holdout_boundary(candidate, holdout_start="2026-01-01")
    with pytest.raises(RuntimeError, match="overlaps holdout"):
        assert_holdout_boundary(candidate, holdout_start="2025-12-31")


def test_holdout_data_workflow_preserves_20_shards_and_pin():
    workflow = load_workflow(Path(".github/workflows/v5-2026-holdout-data.yml"))
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "DATA_END") == "20260904"
    assert max_parallel(workflow, "shard") == 5
    assert matrix_values(workflow, "shard", "shard") == [str(i) for i in range(20)]
    install = normalized_run(workflow, "shard", "Install holdout data dependencies")
    audit = normalized_run(workflow, "merge-and-audit", "Strict 2026 holdout data audit")
    assert "baostock==0.9.3" in install
    assert "--min-symbol-coverage 0.98" in audit
    assert "--min-session-coverage 0.97" in audit
