from pathlib import Path

import pytest

import run_v5_c7_nested_research as c7
from qmt_quant.core_alpha import CORE_ALPHA_FACTORS, CoreAlphaPolicy
from qmt_quant.workflow_contract import env_value, load_workflow, normalized_run, step, structured_text


ROOT = Path(__file__).resolve().parents[1]


def test_c7_policy_is_opt_in_and_c1_core_only():
    assert CoreAlphaPolicy().stability_weighting is False
    assert CoreAlphaPolicy(stability_weighting=True).include_challengers is False
    assert set(CoreAlphaPolicy(stability_weighting=True).allowed_factors) == set(CORE_ALPHA_FACTORS)


def test_c7_runner_refuses_2026_end():
    c7._assert_pre_2026_only(["--end", "2025-12-31"])
    with pytest.raises(RuntimeError, match="pre-2026 research only"):
        c7._assert_pre_2026_only(["--end", "20260102"])


def test_c7_workflow_reuses_exact_frozen_pre_2026_lineage():
    workflow = load_workflow(ROOT / ".github" / "workflows" / "v5-c7-nested-research.yml")
    assert env_value(workflow, "DATA_END") == "20251231"
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "SOURCE_RUN_ID") == "33811845110"
    assert env_value(workflow, "RECOVERY_RUN_ID") == "33887254974"
    assert env_value(workflow, "EXPOSURE_RUN_ID") == "33963211771"
    assert env_value(workflow, "INDUSTRY_RUN_ID") == "33969253365"
    assert env_value(workflow, "QMT_QUANT_CACHE_ONLY") == "1"
    assert step(workflow, "research", "Require exactly one complete set of 20 historical shard manifests")
    assert step(workflow, "research", "Require exactly 20 strict PIT exposure manifests and strict industry recovery")
    audit = normalized_run(workflow, "research", "Revalidate full historical data before C7 research")
    assert "--min-symbol-coverage 0.98" in audit
    assert "--min-session-coverage 0.97" in audit
    semantic = structured_text(workflow)
    for forbidden in ("33963234789", "33963542253", "33967228798"):
        assert forbidden not in semantic


def test_c7_wrapper_forces_stability_and_no_challengers():
    policy = c7._stability_policy()
    assert policy.stability_weighting is True
    assert policy.include_challengers is False
    assert set(policy.allowed_factors) == set(CORE_ALPHA_FACTORS)
