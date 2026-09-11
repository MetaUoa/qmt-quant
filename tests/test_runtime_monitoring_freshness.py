from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import monitoring.check_runtime as check_runtime


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        frame=pd.DataFrame([{"code": "000001.SZ", "target_weight": 1.0}]),
        strategy_sha256="a" * 64,
        target_file_sha256="b" * 64,
        signal_date="20260911",
        expected_execution_session="20260912",
        expires_after_session="20260912",
    )


def _binding() -> dict[str, object]:
    return {
        "batch_id": "c" * 64,
        "account_key": "d" * 64,
        "strategy_sha256": "a" * 64,
        "target_file_sha256": "b" * 64,
        "signal_date": "20260911",
        "expected_execution_session": "20260912",
        "expires_after_session": "20260912",
    }


def _run_monitor(
    tmp_path: Path,
    monkeypatch,
    *,
    broker_health: dict[str, object] | None,
    include_freshness: bool = True,
) -> tuple[int, dict[str, object]]:
    bundle = _bundle()
    monkeypatch.setattr(check_runtime, "validate_target_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(
        check_runtime,
        "validate_acceptance_for_strategy",
        lambda *_a, **_k: {"grade": "A", "schema": "qmt-acceptance-v3"},
    )
    monkeypatch.setattr(check_runtime, "china_market_date", lambda: "2026-09-12")
    monkeypatch.setattr(check_runtime, "runtime_health_alert", lambda _checks: None)

    pretrade = tmp_path / "pretrade.json"
    runtime = tmp_path / "runtime.json"
    freshness = tmp_path / "freshness.json"
    output = tmp_path / "health.json"
    pretrade.write_text(json.dumps({"passed": True}), encoding="utf-8")
    runtime.write_text(json.dumps({"passed": True, "binding": _binding()}), encoding="utf-8")
    if include_freshness:
        freshness.write_text(
            json.dumps(
                {
                    "passed": True,
                    "binding": _binding(),
                    "broker_health": broker_health,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_runtime.py",
            "--acceptance",
            str(tmp_path / "acceptance.json"),
            "--targets",
            str(tmp_path / "targets.csv"),
            "--signal",
            str(tmp_path / "signal.json"),
            "--execution",
            str(pretrade),
            "--runtime-risk",
            str(runtime),
            "--freshness",
            str(freshness),
            "--output",
            str(output),
            "--alert-jsonl",
            str(tmp_path / "alerts.jsonl"),
        ],
    )
    rc = check_runtime.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    return rc, payload


def test_monitor_passes_only_with_connected_drained_callback_state(tmp_path: Path, monkeypatch) -> None:
    rc, payload = _run_monitor(
        tmp_path,
        monkeypatch,
        broker_health={
            "connected": True,
            "connection_lost": False,
            "event_sink_attached": True,
            "event_sink_failed": False,
            "buffered_event_count": 0,
        },
    )
    assert rc == 0
    assert payload["checks"]["passed"] is True


def test_monitor_fails_missing_freshness_artifact(tmp_path: Path, monkeypatch) -> None:
    rc, payload = _run_monitor(
        tmp_path,
        monkeypatch,
        broker_health=None,
        include_freshness=False,
    )
    assert rc == 2
    assert payload["checks"]["freshness_present"] is False
    assert payload["checks"]["passed"] is False


def test_monitor_fails_disconnect_sink_error_unattached_or_undrained_buffer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bad_states = [
        {
            "connected": False,
            "connection_lost": False,
            "event_sink_attached": True,
            "event_sink_failed": False,
            "buffered_event_count": 0,
        },
        {
            "connected": True,
            "connection_lost": True,
            "event_sink_attached": True,
            "event_sink_failed": False,
            "buffered_event_count": 0,
        },
        {
            "connected": True,
            "connection_lost": False,
            "event_sink_attached": False,
            "event_sink_failed": False,
            "buffered_event_count": 0,
        },
        {
            "connected": True,
            "connection_lost": False,
            "event_sink_attached": True,
            "event_sink_failed": True,
            "buffered_event_count": 0,
        },
        {
            "connected": True,
            "connection_lost": False,
            "event_sink_attached": True,
            "event_sink_failed": False,
            "buffered_event_count": 1,
        },
    ]
    for index, state in enumerate(bad_states):
        case = tmp_path / str(index)
        case.mkdir()
        rc, payload = _run_monitor(case, monkeypatch, broker_health=state)
        assert rc == 2
        assert payload["checks"]["passed"] is False
