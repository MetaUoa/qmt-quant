from pathlib import Path


def test_v5_oos_workflow_uses_numpy_safe_entrypoint_and_tracks_runner_changes():
    text = Path(".github/workflows/v5-composite-oos.yml").read_text(encoding="utf-8")

    assert 'python run_v5_composite_oos_entry.py' in text
    assert '- "run_v5_composite_oos.py"' in text
    assert '- "run_v5_composite_oos_entry.py"' in text
    assert 'SHARD_COUNT: "20"' in text
    assert 'SOURCE_RUN_ID: "33811845110"' in text
    assert 'RECOVERY_RUN_ID: "33887254974"' in text
    assert '--min-symbol-coverage 0.98' in text
    assert '--min-session-coverage 0.97' in text
