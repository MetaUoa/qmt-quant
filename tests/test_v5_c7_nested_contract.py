from pathlib import Path

import pytest

import run_v5_c7_nested_research as c7
from qmt_quant.core_alpha import CORE_ALPHA_FACTORS, CoreAlphaPolicy


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
    text = (ROOT / ".github" / "workflows" / "v5-c7-nested-research.yml").read_text(encoding="utf-8")
    assert 'DATA_END: "20251231"' in text
    assert 'SHARD_COUNT: "20"' in text
    assert 'SOURCE_RUN_ID: "33811845110"' in text
    assert 'RECOVERY_RUN_ID: "33887254974"' in text
    assert 'EXPOSURE_RUN_ID: "33963211771"' in text
    assert 'INDUSTRY_RUN_ID: "33969253365"' in text
    assert 'QMT_QUANT_CACHE_ONLY: "1"' in text
    assert "--min-symbol-coverage 0.98" in text
    assert "--min-session-coverage 0.97" in text
    assert "Expected exactly 20 historical shard manifests" in text
    assert "Expected exactly 20 PIT exposure manifests" in text
    assert "33963234789" not in text
    assert "33963542253" not in text
    assert "33967228798" not in text
    assert "2026" not in text.replace("without accessing 2026 holdout", "")


def test_c7_wrapper_forces_stability_and_no_challengers():
    policy = c7._stability_policy()
    assert policy.stability_weighting is True
    assert policy.include_challengers is False
    assert set(policy.allowed_factors) == set(CORE_ALPHA_FACTORS)
