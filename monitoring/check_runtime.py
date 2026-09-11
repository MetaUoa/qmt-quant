from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Mapping

from monitoring.alerts import JsonlAlertSink, runtime_health_alert
from qmt_quant.live_safety import (
    china_market_date,
    validate_acceptance_for_strategy,
    validate_target_bundle,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Runtime artifact health check for QMT Quant")
    p.add_argument("--acceptance", default="output/v5_acceptance/acceptance_report.json")
    p.add_argument("--targets", default="output/live_targets/target_weights.csv")
    p.add_argument("--signal", default="output/live_targets/signal_diagnostics.json")
    p.add_argument("--execution", default="output/live_execution/pretrade_risk.json")
    p.add_argument("--runtime-risk", default="output/live_execution/runtime_risk.json")
    p.add_argument("--freshness", default="output/live_execution/freshness_report.json")
    p.add_argument("--output", default="output/monitoring/runtime_health.json")
    p.add_argument("--alert-jsonl", default="output/monitoring/runtime_alerts.jsonl")
    return p.parse_args()


def _binding_matches_bundle(binding: Mapping[str, object], bundle: object) -> tuple[bool, str, str]:
    batch_id = str(binding.get("batch_id", ""))
    account_key = str(binding.get("account_key", ""))
    matches = bool(
        _SHA256_RE.fullmatch(batch_id)
        and _SHA256_RE.fullmatch(account_key)
        and str(binding.get("strategy_sha256", "")) == str(getattr(bundle, "strategy_sha256"))
        and str(binding.get("target_file_sha256", "")) == str(getattr(bundle, "target_file_sha256"))
        and str(binding.get("signal_date", "")) == str(getattr(bundle, "signal_date"))
        and str(binding.get("expected_execution_session", ""))
        == str(getattr(bundle, "expected_execution_session"))
        and str(binding.get("expires_after_session", "")) == str(getattr(bundle, "expires_after_session"))
    )
    return matches, batch_id, account_key


def main() -> int:
    args = parse_args()
    checks: dict[str, object] = {"market_date": str(china_market_date())}

    bundle = None
    try:
        bundle = validate_target_bundle(
            args.targets,
            args.signal,
            require_current_session=True,
        )
        checks["target_bundle_valid"] = True
        checks["target_count"] = int(len(bundle.frame))
        checks["target_weight_sum"] = float(bundle.frame["target_weight"].sum())
        checks["signal_date"] = str(bundle.signal_date)
        checks["expected_execution_session"] = str(bundle.expected_execution_session)
        checks["expires_after_session"] = str(bundle.expires_after_session)
        checks["strategy_sha256"] = bundle.strategy_sha256
        checks["target_file_sha256"] = bundle.target_file_sha256
    except Exception as exc:
        checks["target_bundle_valid"] = False
        checks["target_bundle_error"] = f"{type(exc).__name__}: {exc}"

    if bundle is not None:
        try:
            acceptance = validate_acceptance_for_strategy(
                args.acceptance,
                "C",
                bundle.strategy_sha256,
            )
            checks["acceptance_ok"] = True
            checks["acceptance_grade"] = acceptance.get("grade")
            checks["acceptance_schema"] = acceptance.get("schema")
        except Exception as exc:
            checks["acceptance_ok"] = False
            checks["acceptance_error"] = f"{type(exc).__name__}: {exc}"
    else:
        checks["acceptance_ok"] = False

    execution_path = Path(args.execution)
    if execution_path.exists():
        risk = json.loads(execution_path.read_text(encoding="utf-8"))
        checks["pretrade_risk_passed"] = bool(risk.get("passed")) if isinstance(risk, dict) else False
    else:
        checks["pretrade_risk_passed"] = False

    runtime_path = Path(args.runtime_risk)
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        checks["runtime_risk_present"] = isinstance(runtime, dict)
        checks["runtime_risk_passed"] = bool(runtime.get("passed")) if isinstance(runtime, dict) else False
        binding = runtime.get("binding") if isinstance(runtime, dict) else None
        if bundle is not None and isinstance(binding, dict):
            matches, batch_id, account_key = _binding_matches_bundle(binding, bundle)
            checks["runtime_binding_match"] = matches
            checks["runtime_batch_id"] = batch_id
            checks["runtime_account_key"] = account_key
        else:
            checks["runtime_binding_match"] = False
    else:
        checks["runtime_risk_present"] = False
        checks["runtime_risk_passed"] = False
        checks["runtime_binding_match"] = False

    freshness_path = Path(args.freshness)
    if freshness_path.exists():
        try:
            freshness = json.loads(freshness_path.read_text(encoding="utf-8"))
        except Exception as exc:
            freshness = None
            checks["freshness_error"] = f"{type(exc).__name__}: {exc}"
        checks["freshness_present"] = isinstance(freshness, dict)
        checks["freshness_passed"] = bool(freshness.get("passed")) if isinstance(freshness, dict) else False
        freshness_binding = freshness.get("binding") if isinstance(freshness, dict) else None
        broker_health = freshness.get("broker_health") if isinstance(freshness, dict) else None
        if bundle is not None and isinstance(freshness_binding, dict):
            matches, batch_id, account_key = _binding_matches_bundle(freshness_binding, bundle)
            checks["freshness_binding_match"] = matches
            checks["freshness_batch_id"] = batch_id
            checks["freshness_account_key"] = account_key
        else:
            checks["freshness_binding_match"] = False
        if isinstance(broker_health, dict):
            checks["broker_connection_not_lost"] = broker_health.get("connection_lost") is False
            checks["broker_event_sink_healthy"] = broker_health.get("event_sink_failed") is False
        else:
            checks["broker_connection_not_lost"] = False
            checks["broker_event_sink_healthy"] = False
    else:
        checks["freshness_present"] = False
        checks["freshness_passed"] = False
        checks["freshness_binding_match"] = False
        checks["broker_connection_not_lost"] = False
        checks["broker_event_sink_healthy"] = False

    mandatory = (
        "target_bundle_valid",
        "acceptance_ok",
        "pretrade_risk_passed",
        "runtime_risk_present",
        "runtime_risk_passed",
        "runtime_binding_match",
        "freshness_present",
        "freshness_passed",
        "freshness_binding_match",
        "broker_connection_not_lost",
        "broker_event_sink_healthy",
    )
    checks["passed"] = all(checks.get(key) is True for key in mandatory)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    alert = runtime_health_alert(checks)
    if alert is not None:
        JsonlAlertSink(args.alert_jsonl).emit(alert)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if checks["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
