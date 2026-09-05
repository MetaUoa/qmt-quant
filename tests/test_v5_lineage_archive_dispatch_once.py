from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "v5-pre2026-lineage-archive-dispatch-once.yml"


def test_dispatcher_can_only_trigger_the_pre2026_archive():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions: write" in text
    assert "gh workflow run v5-pre2026-lineage-archive.yml" in text
    assert "--ref main" in text
    assert "dispatch pre2026 lineage archive once" in text
    assert "schedule:" not in text
    assert "workflow_dispatch:" not in text
    assert "download-artifact" not in text
    for forbidden in (
        "33963234789",
        "33963542253",
        "33967228798",
        "qmt-2026-holdout-data",
        "holdout-2026-exposure",
        "baostock",
        "tushare",
    ):
        assert forbidden not in text.lower()
