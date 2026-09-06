from __future__ import annotations

from pathlib import Path

from qmt_quant.workflow_contract import (
    env_value,
    load_workflow,
    normalized_run,
    structured_text,
    workflow_events,
)


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/v5-c9-neutralization-diagnostics.yml"


def test_c9_remains_manual_only_and_read_only() -> None:
    workflow = load_workflow(WORKFLOW)
    events = workflow_events(workflow)
    assert set(events) == {"workflow_dispatch"}
    assert "push" not in events
    assert "schedule" not in events

    permissions = workflow.get("permissions")
    assert permissions == {"actions": "read", "contents": "read"}


def test_c9_is_pinned_to_frozen_pre_2026_lineage() -> None:
    workflow = load_workflow(WORKFLOW)
    assert env_value(workflow, "DATA_START") == "20170101"
    assert env_value(workflow, "TRADE_START") == "20180101"
    assert env_value(workflow, "DATA_END") == "20251231"
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "SOURCE_RUN_ID") == "33811845110"
    assert env_value(workflow, "RECOVERY_RUN_ID") == "33887254974"
    assert env_value(workflow, "EXPOSURE_RUN_ID") == "33963211771"
    assert env_value(workflow, "INDUSTRY_RUN_ID") == "33969253365"
    assert env_value(workflow, "QMT_QUANT_CACHE_ONLY") == "1"

    text = structured_text(workflow)
    assert "2026" not in text
    assert "holdout" not in text.lower()


def test_c9_runner_keeps_strict_historical_thresholds_and_no_promotion_action() -> None:
    workflow = load_workflow(WORKFLOW)
    command = normalized_run(
        workflow,
        "diagnostics",
        "Run strict purged C1 nested path with fold-safe C9 diagnostics",
    )
    assert "run_v5_c9_neutralization_diagnostics.py" in command
    assert "--end 20251231" in command
    assert "--min-symbol-coverage 0.98" in command
    assert "--min-exposure-coverage 0.95" in command
    assert "--min-symbols-per-date 50" in command

    text = structured_text(workflow).lower()
    assert "production_candidate" not in text
    assert "unlock" not in text
    assert "acceptance" not in text
