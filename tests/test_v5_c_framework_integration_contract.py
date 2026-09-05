from pathlib import Path

from qmt_quant.core_alpha import CORE_ALPHA_FACTORS, CoreAlphaPolicy


RESEARCH_ONLY_MODULES = (
    "alpha_stability",
    "challenger_contracts",
    "neutralization_diagnostics",
    "portfolio_research",
)


def test_c7_c10_frameworks_do_not_modify_current_c1_nested_path():
    research = Path("run_v5_c_nested_research.py").read_text(encoding="utf-8")
    for module in RESEARCH_ONLY_MODULES:
        assert f"qmt_quant.{module}" not in research
        assert f"from qmt_quant.{module}" not in research
    assert CoreAlphaPolicy().allowed_factors == CORE_ALPHA_FACTORS


def test_current_c1_nested_workflow_does_not_reference_holdout_inputs():
    workflow = Path(".github/workflows/v5-c-nested-research.yml").read_text(encoding="utf-8")
    forbidden_holdout_inputs = (
        "33963234789",
        "33963542253",
        "33967228798",
        "qmt-2026-holdout-data",
        "holdout-2026-exposure-shard-12-recovery",
        "holdout-2026-industry-snapshots",
    )
    for token in forbidden_holdout_inputs:
        assert token not in workflow
    assert "without accessing 2026 holdout" in workflow
