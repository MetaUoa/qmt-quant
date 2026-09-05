from pathlib import Path


def test_one_shot_nested_dispatch_is_pinned_and_blinded():
    text = Path(".github/workflows/v5-c-nested-dispatch-once.yml").read_text(encoding="utf-8")
    assert "name: v5-c-nested-dispatch-once" in text
    assert "actions: write" in text
    assert 'INDUSTRY_RUN_ID: "33969253365"' in text
    assert "v5-c-nested-research.yml/dispatches" in text
    assert '\"ref\":\"main\"' in text
    assert '\"industry_run_id\"' in text
    assert "33963211771" not in text
    assert "33963542253" not in text
    assert "33963234789" not in text
    assert "holdout-2026" not in text
    assert "qmt-2026-holdout-data" not in text


def test_one_shot_dispatch_does_not_change_nested_event_model():
    nested = Path(".github/workflows/v5-c-nested-research.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in nested
    assert "workflow_run:" in nested
    assert "- v5-pit-industry-recovery" in nested
    assert "push:" not in nested
