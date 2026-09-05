from __future__ import annotations

import json

from monitoring.alerts import AlertRecord, JsonlAlertSink, runtime_health_alert


def test_jsonl_alert_sink_writes_structured_record(tmp_path):
    path = tmp_path / "alerts.jsonl"
    sink = JsonlAlertSink(path)
    sink.emit(AlertRecord("ERROR", "unit", "failed", {"x": 1}))
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["severity"] == "ERROR"
    assert payload["code"] == "unit"
    assert payload["details"] == {"x": 1}
    assert payload["generated_at_utc"]


def test_runtime_health_alert_is_silent_when_passed():
    assert runtime_health_alert({"passed": True}) is None


def test_runtime_health_alert_lists_failed_boolean_checks():
    alert = runtime_health_alert(
        {
            "passed": False,
            "acceptance_ok": False,
            "signal_present": True,
            "signal_fresh": False,
            "pretrade_risk_passed": False,
        }
    )
    assert alert is not None
    assert alert.code == "runtime_health_failed"
    assert "acceptance_ok" in alert.details["failed_checks"]
    assert "pretrade_risk_passed" in alert.details["failed_checks"]
