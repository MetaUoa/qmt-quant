from pathlib import Path

from qmt_quant.workflow_contract import (
    env_value,
    load_workflow,
    normalized_run,
    step,
    structured_text,
)


def test_v5_composite_oos_reuses_frozen_evidence_and_strict_guards():
    workflow = load_workflow(Path(".github/workflows/v5-composite-oos.yml"))
    assert workflow.get("on") == {"workflow_dispatch": None}
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "SOURCE_RUN_ID") == "33811845110"
    assert env_value(workflow, "RECOVERY_RUN_ID") == "33887254974"
    assert env_value(workflow, "FACTOR_RUN_ID") == "33954426511"
    assert env_value(workflow, "DATA_END") == "20251231"
    assert env_value(workflow, "QMT_QUANT_CACHE_ONLY") == "1"
    for name in (
        "Require exactly one complete set of 20 shard artifacts",
        "Remove stale shard 13 from source run",
        "Download recovered shard 13",
        "Download frozen full-market factor evidence",
    ):
        assert step(workflow, "v5-composite-oos", name)
    audit = normalized_run(
        workflow, "v5-composite-oos", "Revalidate full historical data before V5 OOS"
    )
    runner = normalized_run(
        workflow, "v5-composite-oos", "Run strict purged training-only V5 composite OOS"
    )
    assert "--min-symbol-coverage 0.98" in audit
    assert "--min-session-coverage 0.97" in audit
    assert "run_v5_composite_canonical_oos.py" in runner
    assert "run_v5_composite_oos.py" not in runner
    semantic = structured_text(workflow).lower()
    assert "prepare_free_data_shard.py" not in semantic
    assert "prepare_free_data.py" not in semantic
    assert "baostock" not in semantic
    assert "live" not in semantic


def test_v5_oos_runner_explicitly_uses_purged_training_and_strict_execution():
    text = Path("run_v5_composite_oos.py").read_text(encoding="utf-8")
    assert "select_purged_folds" in text
    assert "max_forward_horizon" in text
    assert "strict_reference=True" in text
    assert "score_override=score" in text
    assert "risk_on_override=risk_on" in text
    assert '"stock_selection_only": True' in text
    assert '"timing_override": "always_on"' in text
