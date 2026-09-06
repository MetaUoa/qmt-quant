from pathlib import Path

from qmt_quant.workflow_contract import env_value, load_workflow, normalized_run, workflow_events


def test_v5_oos_workflow_is_manual_only_after_canonical_cutover():
    workflow = load_workflow(Path(".github/workflows/v5-composite-oos.yml"))
    events = workflow_events(workflow)
    assert "workflow_dispatch" in events
    assert "push" not in events
    runner = normalized_run(
        workflow, "v5-composite-oos", "Run strict purged training-only V5 composite OOS"
    )
    audit = normalized_run(
        workflow, "v5-composite-oos", "Revalidate full historical data before V5 OOS"
    )
    assert "python run_v5_composite_canonical_oos.py" in runner
    assert "python run_v5_composite_oos.py" not in runner
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "SOURCE_RUN_ID") == "33811845110"
    assert env_value(workflow, "RECOVERY_RUN_ID") == "33887254974"
    assert "--min-symbol-coverage 0.98" in audit
    assert "--min-session-coverage 0.97" in audit
