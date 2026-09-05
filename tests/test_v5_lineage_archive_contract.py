from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research_lineage" / "v5_c_pre2026.json"
WORKFLOW = ROOT / ".github" / "workflows" / "v5-pre2026-lineage-archive.yml"


def test_pre2026_lineage_lock_is_exact_and_holdout_blind():
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    assert payload["schema"] == "qmt-v5-lineage-lock-v1"
    assert payload["research_data_end"] == "2025-12-31"
    assert payload["holdout_included"] is False
    assert payload["historical_bars"]["source_run_id"] == 33811845110
    assert payload["historical_bars"]["source_artifact_count"] == 20
    assert payload["historical_bars"]["replace_shard_index"] == 13
    assert payload["historical_bars"]["recovery_run_id"] == 33887254974
    assert payload["pit_exposures"]["run_id"] == 33963211771
    assert payload["pit_exposures"]["artifact_count"] == 20
    assert payload["pit_industry"]["run_id"] == 33969253365
    assert payload["pit_industry"]["recovery_shard_count"] == 12
    assert payload["pit_industry"]["upstream_exposure_run_id"] == 33963211771
    assert payload["invariants"]["max_parallel"] == 5
    assert payload["invariants"]["baostock_version"] == "0.9.3"
    assert payload["invariants"]["min_symbol_coverage"] == 0.98
    assert payload["invariants"]["min_session_coverage"] == 0.97


def test_archive_workflow_is_manual_only_and_never_reacquires_data():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "workflow_run:" not in text
    assert "push:" not in text
    assert 'QMT_QUANT_CACHE_ONLY: "1"' in text
    assert "download-artifact@v4" in text
    assert "download_daily_history" not in text
    assert "baostock" not in text.lower()
    assert "gh release create" in text
    assert "--clobber" not in text


def test_archive_contract_contains_no_2026_holdout_lineage():
    combined = LOCK.read_text(encoding="utf-8") + WORKFLOW.read_text(encoding="utf-8")
    for forbidden in (
        "33963234789",
        "33963542253",
        "33967228798",
        "qmt-2026-holdout-data",
        "holdout-2026-exposure",
    ):
        assert forbidden not in combined
