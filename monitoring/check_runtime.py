from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

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
    p.add_argument("--output", default="output/monitoring/runtime_health.json")
    p.add_argument("--alert-jsonl", default="output/monitoring/runtime_alerts.jsonl")
    return p.parse_args()


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
            batch_id = str(binding.get("batch_id", ""))
            account_key = str(binding.get("account_key", ""))
            checks["runtime_binding_match"] = bool(
                _SHA256_RE.fullmatch(batch_id)
                and _SHA256_RE.fullmatch(account_key)
                and str(binding.get("strategy_sha256", "")) == bundle.strategy_sha256
                and str(binding.get("target_file_sha256", "")) == bundle.target_file_sha256
                and str(binding.get("signal_date", "")) == str(bundle.signal_date)
                and str(binding.get("expected_execution_session", ""))
                == str(bundle.expected_execution_session)
                and str(binding.get("expires_after_session", "")) == str(bundle.expires_after_session)
            )
            checks["runtime_batch_id"] = batch_id
            checks["runtime_account_key"] = account_key
        else:
            checks["runtime_binding_match"] = False
    else:
        checks["runtime_risk_present"] = False
        checks["runtime_risk_passed"] = False
        checks["runtime_binding_match"] = False

    mandatory = (
        "target_bundle_valid",
        "acceptance_ok",
        "pretrade_risk_passed",
        "runtime_risk_present",
        "runtime_risk_passed",
        "runtime_binding_match",
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
