from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from monitoring.alerts import JsonlAlertSink, runtime_health_alert


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


def _market_date() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def main() -> int:
    args = parse_args()
    checks: dict[str, object] = {}
    acceptance: dict = {}
    target_sha = ""

    acceptance_path = Path(args.acceptance)
    if acceptance_path.exists():
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        checks["acceptance_grade"] = acceptance.get("grade")
        checks["acceptance_ok"] = acceptance.get("grade") in {"A", "B", "C"}
    else:
        checks["acceptance_ok"] = False
        checks["acceptance_grade"] = None

    target_path = Path(args.targets)
    if target_path.exists():
        targets = pd.read_csv(target_path)
        checks["targets_present"] = True
        checks["target_count"] = int(len(targets))
        if "target_weight" in targets:
            weights = pd.to_numeric(targets["target_weight"], errors="coerce")
            checks["target_weight_sum"] = float(weights.fillna(0.0).sum())
            checks["target_weights_valid"] = bool(
                weights.notna().all()
                and weights.between(0.0, 1.0).all()
                and float(weights.sum()) <= 1.000001
            )
        else:
            checks["target_weights_valid"] = False
        if "strategy_sha256" in targets and len(targets):
            shas = sorted(set(targets["strategy_sha256"].dropna().astype(str)))
            checks["target_strategy_sha_unique"] = len(shas) == 1
            target_sha = shas[0] if len(shas) == 1 else ""
        else:
            checks["target_strategy_sha_unique"] = False
    else:
        checks["targets_present"] = False
        checks["target_count"] = 0
        checks["target_weights_valid"] = False
        checks["target_strategy_sha_unique"] = False

    signal_path = Path(args.signal)
    if signal_path.exists():
        signal = json.loads(signal_path.read_text(encoding="utf-8"))
        signal_date = signal.get("signal_date")
        checks["signal_date"] = signal_date
        checks["market_date"] = _market_date()
        checks["signal_fresh"] = str(signal_date) == str(checks["market_date"])
        checks["risk_on"] = signal.get("risk_on")
        checks["signal_present"] = True
        source = signal.get("strategy_source") or {}
        signal_sha = str(source.get("sha256", "")) if isinstance(source, dict) else ""
        checks["signal_target_sha_match"] = bool(signal_sha and target_sha and signal_sha == target_sha)
    else:
        checks["signal_present"] = False
        checks["signal_fresh"] = False
        checks["signal_target_sha_match"] = False

    acceptance_sha = str(acceptance.get("strategy_sha256", ""))
    checks["acceptance_target_sha_match"] = bool(
        acceptance_sha and target_sha and acceptance_sha == target_sha
    )

    execution_path = Path(args.execution)
    if execution_path.exists():
        risk = json.loads(execution_path.read_text(encoding="utf-8"))
        checks["pretrade_risk_passed"] = bool(risk.get("passed"))
    else:
        checks["pretrade_risk_passed"] = False

    runtime_path = Path(args.runtime_risk)
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        checks["runtime_risk_passed"] = bool(runtime.get("passed"))
    else:
        checks["runtime_risk_passed"] = None

    mandatory = (
        "acceptance_ok",
        "targets_present",
        "target_weights_valid",
        "target_strategy_sha_unique",
        "signal_present",
        "signal_fresh",
        "signal_target_sha_match",
        "acceptance_target_sha_match",
        "pretrade_risk_passed",
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
