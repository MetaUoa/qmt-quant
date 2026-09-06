from pathlib import Path


def test_v5_composite_workflow_is_manual_only_and_canonical() -> None:
    text = Path('.github/workflows/v5-composite-oos.yml').read_text(encoding='utf-8')
    assert 'workflow_dispatch:' in text
    assert '\n  push:' not in text
    assert 'python run_v5_composite_canonical_oos.py' in text
    assert 'python run_v5_composite_oos.py' not in text
    assert 'SOURCE_RUN_ID: "33811845110"' in text
    assert 'RECOVERY_RUN_ID: "33887254974"' in text
    assert 'FACTOR_RUN_ID: "33954426511"' in text
    assert 'DATA_END: "20251231"' in text
    assert '--min-symbol-coverage 0.98' in text
    assert '--min-session-coverage 0.97' in text
