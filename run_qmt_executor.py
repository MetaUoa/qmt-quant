from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

from qmt_quant.live_safety import validate_acceptance_for_strategy, validate_target_bundle
from qmt_quant.live_trader import QmtBroker, build_equal_weight_plan, serialize_plan
from risk.pretrade import validate_pretrade


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6/V7 MiniQMT rebalance executor; dry-run by default")
    p.add_argument("--userdata", required=True, help="MiniQMT userdata_mini directory")
    p.add_argument("--account", required=True)
    p.add_argument("--account-type", default="STOCK")
    p.add_argument("--session-id", type=int, default=26090201)
    p.add_argument("--targets", default="output/live_targets/target_weights.csv")
    p.add_argument("--target-diagnostics", default="output/live_targets/signal_diagnostics.json")
    p.add_argument("--acceptance", default="output/v5_acceptance/acceptance_report.json")
    p.add_argument("--min-live-grade", choices=["A", "B", "C"], default="C")
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--output", default="output/live_execution")
    p.add_argument("--enable-live", action="store_true")
    p.add_argument("--confirm-live", default="", help="Live mode requires exact value LIVE")
    p.add_argument("--ignore-acceptance", action="store_true", help="Deprecated; live mode refuses this bypass")
    return p.parse_args()


def _append_jsonl_fsync(path: Path, payload: dict) -> None:
    record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    bundle = validate_target_bundle(
        args.targets,
        args.target_diagnostics,
        require_current_session=bool(args.enable_live),
    )
    targets = bundle.frame
    target_codes = [str(x) for x in targets["code"].dropna().tolist()]

    if args.enable_live:
        if args.confirm_live != "LIVE":
            raise RuntimeError("Live execution requires --confirm-live LIVE")
        if args.ignore_acceptance:
            raise RuntimeError("Live acceptance bypass is disabled")
        validate_acceptance_for_strategy(
            args.acceptance,
            args.min_live_grade,
            bundle.strategy_sha256,
        )

    broker = QmtBroker(args.userdata, args.account, args.session_id, args.account_type)
    broker.connect(max_attempts=3, retry_delay_seconds=1.0)
    total_asset, cash, positions = broker.snapshot()
    all_codes = list(dict.fromkeys(target_codes + list(positions)))
    ticks = broker.full_tick(all_codes)
    executable = broker.executable_prices(ticks)
    prices = {code: float(v["last"] or v["buy"] or v["sell"]) for code, v in executable.items()}
    plan = build_equal_weight_plan(
        target_codes,
        prices,
        positions,
        total_asset=total_asset,
        exposure=args.exposure,
        lot_size=100,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "total_asset": total_asset,
        "cash": cash,
        "position_count": len(positions),
        "target_count": len(target_codes),
        "signal_date": str(bundle.signal_date),
        "strategy_sha256": bundle.strategy_sha256,
        "dry_run": not args.enable_live,
    }
    (out / "pretrade_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(serialize_plan(plan)).to_csv(out / "order_plan.csv", index=False, encoding="utf-8-sig")
    risk_report = validate_pretrade(plan, total_asset=total_asset, target_count=len(target_codes))
    (out / "pretrade_risk.json").write_text(json.dumps(risk_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    print(json.dumps({"pretrade_risk": risk_report}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(serialize_plan(plan)).to_string(index=False) if plan else "No orders required")

    if not args.enable_live:
        print("DRY RUN: no orders were sent")
        return 0
    if not risk_report["passed"]:
        raise RuntimeError(f"Pre-trade risk gate failed: {risk_report['violations']}")

    journal = out / "order_journal.jsonl"
    _append_jsonl_fsync(
        journal,
        {
            "event": "EXECUTION_START",
            "signal_date": str(bundle.signal_date),
            "strategy_sha256": bundle.strategy_sha256,
            "planned_order_count": len(plan),
        },
    )
    results = broker.submit_plan(plan, on_event=lambda event: _append_jsonl_fsync(journal, event))
    pd.DataFrame(results).to_csv(out / "submitted_orders.csv", index=False, encoding="utf-8-sig")
    (out / "submitted_orders.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    submitted_ids = [int(x.get("order_id", 0) or 0) for x in results if int(x.get("order_id", 0) or 0) > 0]
    reconciliation = broker.reconcile_order_ids(
        submitted_ids,
        max_attempts=3,
        retry_delay_seconds=0.5,
    )
    if any(x.get("status") == "SUBMIT_EXCEPTION" for x in results):
        reconciliation["requires_manual_reconciliation"] = True
        reconciliation["uncertain_submit_exception"] = True
    (out / "order_reconciliation.json").write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(reconciliation.get("orders", [])).to_csv(
        out / "order_reconciliation.csv", index=False, encoding="utf-8-sig"
    )
    _append_jsonl_fsync(journal, {"event": "RECONCILIATION", **reconciliation})

    incomplete = [x for x in results if x.get("status") != "SUBMITTED"]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({"reconciliation": reconciliation}, ensure_ascii=False, indent=2))
    if reconciliation.get("requires_manual_reconciliation"):
        return 4
    return 3 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
