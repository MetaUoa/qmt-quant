from pathlib import Path


def test_c_nested_workflow_reuses_frozen_data_and_strict_exposures():
    text = Path(".github/workflows/v5-c-nested-research.yml").read_text(encoding="utf-8")
    assert 'SHARD_COUNT: "20"' in text
    assert 'SOURCE_RUN_ID: "33811845110"' in text
    assert 'RECOVERY_RUN_ID: "33887254974"' in text
    assert 'EXPOSURE_RUN_ID: "33963211771"' in text
    assert 'QMT_QUANT_CACHE_ONLY: "1"' in text
    assert "Remove stale shard 13 from source run" in text
    assert "Download recovered shard 13" in text
    assert "Expected exactly 20 historical shard manifests" in text
    assert "Expected exactly 20 PIT exposure manifests" in text
    assert "--min-symbol-coverage 0.98" in text
    assert "--min-session-coverage 0.97" in text
    assert "run_v5_c_nested_research.py" in text
    assert "qmt-2026-holdout-data" not in text
    assert "holdout-2026" not in text
    assert "prepare_free_data_shard.py" not in text


def test_2026_exposure_workflow_is_blinded_and_preserves_sharding():
    text = Path(".github/workflows/v5-2026-holdout-exposures.yml").read_text(encoding="utf-8")
    assert 'SHARD_COUNT: "20"' in text
    assert "max-parallel: 5" in text
    assert 'DATA_START: "20260101"' in text
    assert 'DATA_END: "20260904"' in text
    assert "baostock==0.9.3" in text
    assert "prepare_pit_exposure_shard.py" in text
    assert "prepare_pit_industry.py" in text
    assert "run_backtest" not in text
    assert "run_v5_c_nested_research.py" not in text
    assert "holdout result" not in text.lower()


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
