from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from merge_pit_industry_shards import merge_industry_shards
from prepare_pit_industry import fetch_trade_calendar_with_retry, select_snapshot_shard


def test_snapshot_shards_are_disjoint_complete_and_ordered():
    dates = pd.date_range("2017-01-03", periods=108, freq="MS")
    shards = [select_snapshot_shard(dates, i, 12) for i in range(12)]

    flattened = [ts for shard in shards for ts in shard]
    assert len(flattened) == len(dates)
    assert len(set(flattened)) == len(dates)
    assert set(flattened) == set(dates)
    for shard in shards:
        assert list(shard) == sorted(shard)
        assert len(shard) == 9


def test_calendar_bootstrap_reconnects_then_succeeds(monkeypatch):
    calls = 0
    reconnects: list[float] = []
    expected = pd.DataFrame({"cal_date": ["20240102"], "is_open": [1]})

    def fake_calendar(api, start, end):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("calendar timeout")
        return expected.copy()

    monkeypatch.setattr("prepare_pit_industry.fetch_trade_calendar", fake_calendar)
    monkeypatch.setattr(
        "prepare_pit_industry._reconnect_baostock",
        lambda api, *, socket_timeout_seconds: reconnects.append(socket_timeout_seconds),
    )
    monkeypatch.setattr("prepare_pit_industry.time.sleep", lambda _: None)

    result = fetch_trade_calendar_with_retry(
        object(),
        "20240101",
        "20240131",
        attempts=2,
        sleep_seconds=0.0,
        socket_timeout_seconds=23.0,
    )
    assert calls == 2
    assert reconnects == [23.0]
    pd.testing.assert_frame_equal(result, expected)


def _write_industry_shard(root: Path, index: int, count: int, dates: list[str]) -> None:
    shard_root = root / f"pit-industry-recovery-shard-{index}"
    shard_root.mkdir(parents=True)
    rows = [
        {"asof_date": day, "ts_code": "000001.SZ", "industry": f"I{index}"}
        for day in dates
    ]
    pd.DataFrame(rows).to_parquet(shard_root / "industry_snapshots.parquet", index=False)
    manifest = {
        "source": "baostock-query_stock_industry",
        "start": "20170101",
        "end": "20171231",
        "snapshot_frequency": "monthly_first_open_session",
        "shard_index": index,
        "shard_count": count,
        "snapshot_candidates_total": 4,
        "snapshots_expected": len(dates),
        "snapshots_written": len(dates),
        "rows": len(rows),
        "errors": [],
        "strict_ready": True,
    }
    (shard_root / "industry_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_merge_industry_shards_requires_exact_complete_set(tmp_path: Path):
    input_root = tmp_path / "in"
    _write_industry_shard(input_root, 0, 2, ["2017-01-03", "2017-03-01"])
    _write_industry_shard(input_root, 1, 2, ["2017-02-03", "2017-04-05"])

    out = tmp_path / "out"
    manifest = merge_industry_shards(
        input_root,
        out,
        upstream_exposure_run_id=33963211771,
    )
    assert manifest["strict_ready"] is True
    assert manifest["merged_shards"] == 2
    assert manifest["snapshots_written"] == 4
    assert manifest["upstream_exposure_run_id"] == 33963211771
    merged = pd.read_parquet(out / "industry_snapshots.parquet")
    assert merged["asof_date"].nunique() == 4

    (input_root / "pit-industry-recovery-shard-1" / "industry_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="Missing PIT industry shard indexes"):
        merge_industry_shards(
            input_root,
            tmp_path / "broken",
            upstream_exposure_run_id=33963211771,
        )


def test_industry_recovery_workflow_is_sharded_pinned_and_nested_consumes_it():
    recovery = Path(".github/workflows/v5-pit-industry-recovery.yml").read_text(
        encoding="utf-8"
    )
    nested = Path(".github/workflows/v5-c-nested-research.yml").read_text(
        encoding="utf-8"
    )
    original = Path(".github/workflows/v5-pit-exposures.yml").read_text(encoding="utf-8")

    assert 'INDUSTRY_SHARD_COUNT: "12"' in recovery
    assert 'UPSTREAM_EXPOSURE_RUN_ID: "33963211771"' in recovery
    assert "github.event.workflow_run.id == 33963211771" in recovery
    assert "max-parallel: 5" in recovery
    assert "matrix:" in recovery and "11]" in recovery
    assert "baostock==0.9.3" in recovery
    assert "workflow_run:" in recovery
    assert "- v5-pit-exposures" in recovery
    assert "--upstream-exposure-run-id" in recovery
    assert "pit-industry-snapshots" in recovery

    assert "- v5-pit-industry-recovery" in nested
    assert 'EXPOSURE_RUN_ID: "33963211771"' in nested
    assert "INDUSTRY_RUN_ID:" in nested
    assert "github.event.workflow_run.id" in nested
    assert "Recovered PIT industry snapshots are not strict-ready" in nested
    assert "Recovered PIT industry upstream run mismatch" in nested
    assert "--min-symbol-coverage 0.98" in nested
    assert "--min-session-coverage 0.97" in nested
    assert "qmt-2026-holdout-data" not in nested

    assert "workflow_dispatch:" in original
    assert "push:" not in original
    assert 'SHARD_COUNT: "20"' in original
    assert "max-parallel: 5" in original
