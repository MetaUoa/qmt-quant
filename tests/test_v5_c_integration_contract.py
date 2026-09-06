from pathlib import Path

from qmt_quant.workflow_contract import (
    env_value,
    job,
    load_workflow,
    matrix_values,
    max_parallel,
    normalized_run,
    step,
    structured_text,
    workflow_events,
)


def test_c_nested_workflow_reuses_frozen_data_and_strict_exposures():
    workflow = load_workflow(Path(".github/workflows/v5-c-nested-research.yml"))
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "SOURCE_RUN_ID") == "33811845110"
    assert env_value(workflow, "RECOVERY_RUN_ID") == "33887254974"
    assert env_value(workflow, "EXPOSURE_RUN_ID") == "33963211771"
    assert env_value(workflow, "QMT_QUANT_CACHE_ONLY") == "1"
    events = workflow_events(workflow)
    assert "push" not in events
    assert events["workflow_run"]["workflows"] == ["v5-pit-industry-recovery"]
    assert events["workflow_run"]["types"] == ["completed"]
    assert "github.event.workflow_run.conclusion == 'success'" in str(job(workflow, "research").get("if"))
    for name in (
        "Remove stale shard 13 from source run",
        "Download recovered shard 13",
        "Require exactly one complete set of 20 historical shard manifests",
        "Require exactly 20 strict PIT exposure manifests and strict industry recovery",
    ):
        assert step(workflow, "research", name)
    audit = normalized_run(workflow, "research", "Revalidate full historical data before C research")
    runner = normalized_run(workflow, "research", "Run strict purged nested V5 C research")
    assert "--min-symbol-coverage 0.98" in audit
    assert "--min-session-coverage 0.97" in audit
    assert "run_v5_c_nested_research.py" in runner
    semantic = structured_text(workflow)
    assert "qmt-2026-holdout-data" not in semantic
    assert "holdout-2026" not in semantic
    assert "prepare_free_data_shard.py" not in semantic


def test_2026_exposure_workflow_is_blinded_and_preserves_sharding():
    workflow = load_workflow(Path(".github/workflows/v5-2026-holdout-exposures.yml"))
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "DATA_START") == "20260101"
    assert env_value(workflow, "DATA_END") == "20260904"
    assert max_parallel(workflow, "exposure-shard") == 5
    assert matrix_values(workflow, "exposure-shard", "shard") == [str(i) for i in range(20)]
    exposure_install = normalized_run(workflow, "exposure-shard", "Install exposure dependencies")
    exposure_run = normalized_run(
        workflow, "exposure-shard", "Download deterministic blinded 2026 float-cap exposure sidecar"
    )
    industry_run = normalized_run(
        workflow, "industry", "Download blinded 2026 monthly PIT industry snapshots"
    )
    assert "baostock==0.9.3" in exposure_install
    assert "prepare_pit_exposure_shard.py" in exposure_run
    assert "prepare_pit_industry.py" in industry_run
    semantic = structured_text(workflow).lower()
    assert "run_backtest" not in semantic
    assert "run_v5_c_nested_research.py" not in semantic
    assert "holdout result" not in semantic


def test_c_runner_keeps_stock_selection_only_and_basic_alpha_gate():
    text = Path("run_v5_c_nested_research.py").read_text(encoding="utf-8")
    assert 'VARIANTS = ("raw", "liquidity", "industry", "industry_size_liquidity")' in text
    assert "CoreAlphaPolicy(include_challengers=False)" in text
    assert "strict_reference=True" in text
    assert "risk_on_override=risk_on" in text
    assert '"positive_oos_return"' in text
    assert '"sharpe_at_least_0_5"' in text
    assert '"max_drawdown_at_most_0_35"' in text
    assert '"at_least_4_non_disastrous_folds"' in text
    assert '"holdout_unlocked": bool(gate["passed"])' in text
    assert 'research_data_end="2025-12-31"' in text
