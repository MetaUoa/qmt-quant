from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Runtime artifact health check for QMT Quant V7")
    p.add_argument("--acceptance", default="output/v5_acceptance/acceptance_report.json")
    p.add_argument("--targets", default="output/live_targets/target_weights.csv")
    p.add_argument("--signal", default="output/live_targets/signal_diagnostics.json")
    p.add_argument("--execution", default="output/live_execution/pretrade_risk.json")
    p.add_argument("--output", default="output/monitoring/runtime_health.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    checks: dict[str, object] = {}
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
            checks["target_weight_sum"] = float(pd.to_numeric(targets["target_weight"], errors="coerce").fillna(0.0).sum())
    else:
        checks["targets_present"] = False
        checks["target_count"] = 0

    signal_path = Path(args.signal)
    if signal_path.exists():
        signal = json.loads(signal_path.read_text(encoding="utf-8"))
        checks["signal_date"] = signal.get("signal_date")
        checks["risk_on"] = signal.get("risk_on")
        checks["signal_present"] = True
    else:
        checks["signal_present"] = False

    execution_path = Path(args.execution)
    if execution_path.exists():
        risk = json.loads(execution_path.read_text(encoding="utf-8"))
        checks["pretrade_risk_passed"] = bool(risk.get("passed"))
    else:
        checks["pretrade_risk_passed"] = None

    checks["passed"] = bool(checks.get("acceptance_ok") and checks.get("targets_present") and checks.get("signal_present"))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if checks["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
