from pathlib import Path


def test_v5_composite_oos_reuses_frozen_evidence_and_strict_guards():
    text = Path(".github/workflows/v5-composite-oos.yml").read_text(encoding="utf-8")

    assert 'SHARD_COUNT: "20"' in text
    assert 'SOURCE_RUN_ID: "33811845110"' in text
    assert 'RECOVERY_RUN_ID: "33887254974"' in text
    assert 'FACTOR_RUN_ID: "33954426511"' in text
    assert 'QMT_QUANT_CACHE_ONLY: "1"' in text
    assert "Require exactly one complete set of 20 shard artifacts" in text
    assert "Remove stale shard 13 from source run" in text
    assert "Download recovered shard 13" in text
    assert "Download frozen full-market factor evidence" in text
    assert "--min-symbol-coverage 0.98" in text
    assert "--min-session-coverage 0.97" in text
    assert "run_v5_composite_oos.py" in text
    assert "prepare_free_data_shard.py" not in text
    assert "prepare_free_data.py" not in text
    assert "baostock" not in text.lower()
    assert "live" not in text.lower()


def test_v5_oos_runner_explicitly_uses_purged_training_and_strict_execution():
    text = Path("run_v5_composite_oos.py").read_text(encoding="utf-8")
    assert "select_purged_folds" in text
    assert "max_forward_horizon" in text
    assert "strict_reference=True" in text
    assert "score_override=score" in text
    assert "risk_on_override=risk_on" in text
    assert '"stock_selection_only": True' in text
    assert '"timing_override": "always_on"' in text
