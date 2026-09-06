from pathlib import Path

from qmt_quant.workflow_contract import (
    env_value,
    load_workflow,
    matrix_values,
    max_parallel,
    normalized_run,
)


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "free-data-full-market.yml"
SHARD_SCRIPT = ROOT / "prepare_free_data_shard.py"


def test_full_market_keeps_small_shards_with_five_way_parallelism() -> None:
    workflow = load_workflow(WORKFLOW)
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert max_parallel(workflow, "shard") == 5
    assert matrix_values(workflow, "shard", "shard") == [str(i) for i in range(20)]


def test_full_market_strict_integrity_gates_remain_enabled() -> None:
    workflow = load_workflow(WORKFLOW)
    audit = normalized_run(workflow, "merge-and-research", "Full historical data audit")
    walk_forward = normalized_run(
        workflow, "merge-and-research", "Independent reset-capital walk-forward"
    )
    assert "--min-symbol-coverage 0.98" in audit
    assert "--min-session-coverage 0.97" in audit
    assert "--strict-reference" in walk_forward


def test_shard_script_default_partition_count_matches_workflow() -> None:
    text = SHARD_SCRIPT.read_text(encoding="utf-8")
    assert 'p.add_argument("--shard-count", type=int, default=20)' in text
