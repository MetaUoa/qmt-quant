from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "free-data-full-market.yml"
SHARD_SCRIPT = ROOT / "prepare_free_data_shard.py"


def test_full_market_keeps_small_shards_with_five_way_parallelism() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'SHARD_COUNT: "20"' in text
    assert "max-parallel: 5" in text
    assert "shard: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]" in text


def test_full_market_strict_integrity_gates_remain_enabled() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "--min-symbol-coverage 0.98" in text
    assert "--min-session-coverage 0.97" in text
    assert "--strict-reference" in text


def test_shard_script_default_partition_count_matches_workflow() -> None:
    text = SHARD_SCRIPT.read_text(encoding="utf-8")
    assert 'p.add_argument("--shard-count", type=int, default=20)' in text
