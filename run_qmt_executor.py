from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

from qmt_quant.execution_state import (
    account_execution_key,
    execution_batch_id,
    load_account_execution_lock,
    release_account_execution_lock,
    reserve_account_execution_lock,
    reserve_execution_batch,
    update_execution_batch,
)
from qmt_quant.live_safety import validate_acceptance_for_strategy, validate_target_bundle
from qmt_quant.live_trader import QmtBroker, serialize_plan
from qmt_quant.target_planning import build_target_weight_plan
from risk.pretrade import validate_pretrade
from risk.runtime import RuntimeRiskPolicy, evaluate_runtime_risk


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
    p.add_argument(
        "--exposure",
        type=float,
        default=1.0,
        help="Optional downward scale applied to target weights; never renormalizes them upward",
    )
    p.add_argument(
        "--start-of-day-equity",
        type=float,
        default=None,
        help="Required for live runtime drawdown gating",
    )
    p.add_argument(
        "--runtime-kill-switch",
        action="store_true",
        help="Fail the runtime risk gate before any live order submission",
    )
    p.add_argument("--output", default="output/live_execution")
    p.add_argument(
        "--state-dir",
        default="output/execution_state",
        help="Persistent account locks/batch markers; keep stable across executor output directories",
    )
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _block_for_existing_account_lock(
    broker: QmtBroker,
    *,
    lock_payload: dict,
    output: Path,
) -> None:
    recovery: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_ACTIVE_ACCOUNT_LOCK",
        "active_lock": lock_payload,
        "requires_operator_reconciliation": True,
    }
    try:
        recovery["observed_orders"] = broker.query_orders(cancelable_only=False)
        recovery["observed_trades"] = broker.query_trades()
    except Exception as exc:
        recovery["broker_query_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    _write_json(output / "startup_recovery.json", recovery)
    raise RuntimeError(
        "an active account execution lock already exists; startup reconciliation was recorded "
        "and new orders are blocked pending operator review"
    )


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    state_root = Path(args.state_dir)
    account_key = account_execution_key(account_id=args.account, account_type=args.account_type)

    bundle = validate_target_bundle(
        args.targets,
        args.target_diagnostics,
        require_current_session=bool(args.enable_live),
    )
    targets = bundle.frame
    target_weights = {
        str(row.code): float(row.target_weight)
        for row in targets[["code", "target_weight"]].itertuples(index=False)
    }
    target_codes = list(target_weights)

    if args.enable_live:
        if args.confirm_live != "LIVE":
            raise RuntimeError("Live execution requires --confirm-live LIVE")
        if args.ignore_acceptance:
            raise RuntimeError("Live acceptance bypass is disabled")
        if args.start_of_day_equity is None:
            raise RuntimeError("Live execution requires --start-of-day-equity for runtime risk gating")
        validate_acceptance_for_strategy(
            args.acceptance,
            args.min_live_grade,
            bundle.strategy_sha256,
        )

    broker = QmtBroker(args.userdata, args.account, args.session_id, args.account_type)
    broker.connect(max_attempts=3, retry_delay_seconds=1.0)

    if args.enable_live:
        existing_lock = load_account_execution_lock(state_root, account_key=account_key)
        if existing_lock is not None:
            _block_for_existing_account_lock(broker, lock_payload=existing_lock, output=out)

    total_asset, cash, positions = broker.snapshot()
    all_codes = list(dict.fromkeys(target_codes + list(positions)))
    ticks = broker.full_tick(all_codes)
    executable = broker.executable_prices(ticks)
    prices = {code: float(v["last"] or v["buy"] or v["sell"]) for code, v in executable.items()}
    plan, effective_target_weights = build_target_weight_plan(
        target_weights,
        prices,
        positions,
        total_asset=total_asset,
        exposure=args.exposure,
        lot_size=100,
    )
    batch_id = execution_batch_id(
        signal_date=str(bundle.signal_date),
        strategy_sha256=bundle.strategy_sha256,
        target_weights=effective_target_weights,
        account_key=account_key,
    )

    snapshot = {
        "total_asset": total_asset,
        "cash": cash,
        "position_count": len(positions),
        "target_count": len(target_codes),
        "requested_target_weight_sum": float(sum(target_weights.values())),
        "effective_target_weight_sum": float(sum(effective_target_weights.values())),
        "signal_date": str(bundle.signal_date),
        "expected_execution_session": str(bundle.expected_execution_session),
        "expires_after_session": str(bundle.expires_after_session),
        "strategy_sha256": bundle.strategy_sha256,
        "target_file_sha256": bundle.target_file_sha256,
        "account_key": account_key,
        "batch_id": batch_id,
        "dry_run": not args.enable_live,
    }
    _write_json(out / "pretrade_snapshot.json", snapshot)
    pd.DataFrame(serialize_plan(plan)).to_csv(out / "order_plan.csv", index=False, encoding="utf-8-sig")
    risk_report = validate_pretrade(
        plan,
        total_asset=total_asset,
        target_count=len(target_codes),
        target_weights=effective_target_weights,
    )
    _write_json(out / "pretrade_risk.json", risk_report)

    runtime_report = None
    if args.start_of_day_equity is not None:
        runtime_report = evaluate_runtime_risk(
            start_of_day_equity=args.start_of_day_equity,
            current_equity=total_asset,
            target_codes=target_codes,
            policy=RuntimeRiskPolicy(kill_switch=bool(args.runtime_kill_switch)),
        )
        runtime_report["binding"] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "account_key": account_key,
            "batch_id": batch_id,
            "strategy_sha256": bundle.strategy_sha256,
            "target_file_sha256": bundle.target_file_sha256,
            "signal_date": str(bundle.signal_date),
            "expected_execution_session": str(bundle.expected_execution_session),
            "expires_after_session": str(bundle.expires_after_session),
        }
        _write_json(out / "runtime_risk.json", runtime_report)

    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    print(json.dumps({"pretrade_risk": risk_report}, ensure_ascii=False, indent=2))
    if runtime_report is not None:
        print(json.dumps({"runtime_risk": runtime_report}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(serialize_plan(plan)).to_string(index=False) if plan else "No orders required")

    if not args.enable_live:
        print("DRY RUN: no orders were sent")
        return 0
    if not risk_report["passed"]:
        raise RuntimeError(f"Pre-trade risk gate failed: {risk_report['violations']}")
    if runtime_report is None or not runtime_report["passed"]:
        violations = [] if runtime_report is None else runtime_report["violations"]
        raise RuntimeError(f"Runtime risk gate failed: {violations}")

    batch_marker = reserve_execution_batch(
        state_root / "execution_batches",
        batch_id=batch_id,
        metadata={
            "signal_date": str(bundle.signal_date),
            "expected_execution_session": str(bundle.expected_execution_session),
            "expires_after_session": str(bundle.expires_after_session),
            "strategy_sha256": bundle.strategy_sha256,
            "target_file_sha256": bundle.target_file_sha256,
            "account_key": account_key,
            "account_type": args.account_type,
            "planned_order_count": len(plan),
            "effective_target_weight_sum": float(sum(effective_target_weights.values())),
        },
    )
    try:
        account_lock = reserve_account_execution_lock(
            state_root,
            account_key=account_key,
            batch_id=batch_id,
            metadata={
                "account_type": args.account_type,
                "signal_date": str(bundle.signal_date),
                "expected_execution_session": str(bundle.expected_execution_session),
                "strategy_sha256": bundle.strategy_sha256,
            },
        )
    except Exception as exc:
        update_execution_batch(
            batch_marker,
            status="BLOCKED_ACCOUNT_LOCK",
            details={"error_type": type(exc).__name__},
        )
        raise

    journal = out / "order_journal.jsonl"
    _append_jsonl_fsync(
        journal,
        {
            "event": "EXECUTION_START",
            "batch_id": batch_id,
            "account_key": account_key,
            "signal_date": str(bundle.signal_date),
            "expected_execution_session": str(bundle.expected_execution_session),
            "strategy_sha256": bundle.strategy_sha256,
            "planned_order_count": len(plan),
            "effective_target_weight_sum": float(sum(effective_target_weights.values())),
        },
    )

    try:
        results = broker.submit_plan(
            plan,
            on_event=lambda event: _append_jsonl_fsync(
                journal, {"batch_id": batch_id, "account_key": account_key, **event}
            ),
        )
        pd.DataFrame(results).to_csv(out / "submitted_orders.csv", index=False, encoding="utf-8-sig")
        _write_json(out / "submitted_orders.json", {"orders": results})

        submitted_ids = [
            int(x.get("order_id", 0) or 0)
            for x in results
            if int(x.get("order_id", 0) or 0) > 0
        ]
        if submitted_ids:
            reconciliation = broker.reconcile_order_ids(
                submitted_ids,
                max_attempts=3,
                retry_delay_seconds=0.5,
            )
        else:
            reconciliation = {
                "orders": [],
                "missing_order_ids": [],
                "requires_manual_reconciliation": False,
            }
        if any(x.get("status") == "SUBMIT_EXCEPTION" for x in results):
            reconciliation["requires_manual_reconciliation"] = True
            reconciliation["uncertain_submit_exception"] = True
    except Exception as exc:
        failure = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "batch_id": batch_id,
            "account_key": account_key,
            "status": "MANUAL_RECONCILIATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(out / "execution_failure.json", failure)
        _append_jsonl_fsync(journal, {"event": "EXECUTION_ABORTED_UNKNOWN_STATE", **failure})
        update_execution_batch(
            batch_marker,
            status="MANUAL_RECONCILIATION",
            details=failure,
        )
        # Deliberately retain the account lock. A new process must reconcile broker
        # orders/trades before an operator decides whether the lock can be cleared.
        return 4

    _write_json(out / "order_reconciliation.json", reconciliation)
    pd.DataFrame(reconciliation.get("orders", [])).to_csv(
        out / "order_reconciliation.csv", index=False, encoding="utf-8-sig"
    )
    _append_jsonl_fsync(journal, {"event": "RECONCILIATION", "batch_id": batch_id, **reconciliation})

    incomplete = [x for x in results if x.get("status") != "SUBMITTED"]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({"reconciliation": reconciliation}, ensure_ascii=False, indent=2))
    if reconciliation.get("requires_manual_reconciliation"):
        update_execution_batch(
            batch_marker,
            status="MANUAL_RECONCILIATION",
            details={"reconciliation": reconciliation},
        )
        # Retain account_lock for recovery-first startup behavior.
        return 4
    if incomplete:
        update_execution_batch(
            batch_marker,
            status="INCOMPLETE",
            details={"results": results},
        )
        release_account_execution_lock(account_lock, batch_id=batch_id)
        return 3

    update_execution_batch(
        batch_marker,
        status="COMPLETED",
        details={"submitted_order_ids": submitted_ids},
    )
    release_account_execution_lock(account_lock, batch_id=batch_id)
    _append_jsonl_fsync(
        journal,
        {"event": "EXECUTION_COMPLETE", "batch_id": batch_id, "account_key": account_key},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
