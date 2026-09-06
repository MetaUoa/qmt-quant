from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from prepare_pit_exposure_shard import fetch_active_stock_basic_with_retry
from qmt_quant.workflow_contract import (
    env_value,
    job,
    load_workflow,
    normalized_run,
    step,
    structured_text,
)


def test_stock_basic_bootstrap_reconnects_then_succeeds(monkeypatch):
    api = object()
    calls = 0
    reconnects: list[float] = []
    expected = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "list_date": ["19910403"],
            "delist_date": [""],
        }
    )

    def fake_fetch(_api):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("bootstrap timeout")
        return expected.copy()

    monkeypatch.setattr("prepare_pit_exposure_shard.fetch_stock_basic", fake_fetch)
    monkeypatch.setattr(
        "prepare_pit_exposure_shard.active_in_range",
        lambda frame, start, end: frame,
    )
    monkeypatch.setattr(
        "prepare_pit_exposure_shard._reconnect_baostock",
        lambda _api, *, socket_timeout_seconds: reconnects.append(socket_timeout_seconds),
    )
    monkeypatch.setattr("prepare_pit_exposure_shard.time.sleep", lambda _: None)

    result = fetch_active_stock_basic_with_retry(
        api,
        "20260101",
        "20260904",
        attempts=2,
        sleep_seconds=0.0,
        socket_timeout_seconds=17.0,
    )

    assert calls == 2
    assert reconnects == [17.0]
    pd.testing.assert_frame_equal(result, expected)


def test_stock_basic_bootstrap_fails_closed_after_exhaustion(monkeypatch):
    reconnects: list[float] = []
    monkeypatch.setattr(
        "prepare_pit_exposure_shard.fetch_stock_basic",
        lambda _api: (_ for _ in ()).throw(TimeoutError("bootstrap timeout")),
    )
    monkeypatch.setattr(
        "prepare_pit_exposure_shard._reconnect_baostock",
        lambda _api, *, socket_timeout_seconds: reconnects.append(socket_timeout_seconds),
    )
    monkeypatch.setattr("prepare_pit_exposure_shard.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="stock-basic bootstrap failed after retries"):
        fetch_active_stock_basic_with_retry(
            object(),
            "20260101",
            "20260904",
            attempts=3,
            sleep_seconds=0.0,
            socket_timeout_seconds=19.0,
        )
    assert reconnects == [19.0, 19.0]


def test_shard12_recovery_is_single_shard_utf8_and_pinned():
    workflow = load_workflow(Path(".github/workflows/v5-2026-exposure-shard12-recovery.yml"))
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "PYTHONUTF8") == "1"
    assert env_value(workflow, "PYTHONIOENCODING") == "utf-8"
    assert "matrix" not in job(workflow, "recover-shard-12")
    install = normalized_run(workflow, "recover-shard-12", "Install recovery dependencies")
    recover = normalized_run(
        workflow, "recover-shard-12", "Recover deterministic blinded shard 12 only"
    )
    upload = step(
        workflow, "recover-shard-12", "Upload recovered blinded 2026 PIT exposure shard 12"
    )
    assert "baostock==0.9.3" in install
    assert "--shard-index 12" in recover
    assert "--shard-count $env:SHARD_COUNT" in recover
    assert upload["with"]["name"] == "holdout-2026-exposure-shard-12-recovery"
    semantic = structured_text(workflow).lower()
    assert "run_backtest" not in semantic
    assert "holdout result" not in semantic
