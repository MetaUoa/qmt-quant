from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

from qmt_quant.config import CostConfig
from qmt_quant.execution_phases import (
    estimate_buy_cash_reserve,
    has_uncertain_submission,
    incomplete_results,
    split_order_plan,
    submitted_order_ids,
    validate_sell_position_effect,
)
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


_DEFAULT_COST = CostConfig()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiniQMT rebalance executor; dry-run by default")
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
    p.add_argument("--commission-rate", type=float, default=_DEFAULT_COST.commission_rate)
    p.add_argument("--min-commission", type=float, default=_DEFAULT_COST.min_commission)
    p.add_argument("--lot-size", type=int, default=_DEFAULT_COST.lot_size)
    p.add_argument("--buy-buffer-bps", type=float, default=8.0)
    p.add_argument("--sell-buffer-bps", type=float, default=8.0)
    p.add_argument("--output", default="output/live_execution")
    p.add_argument(
        "--state-dir",
        default="output/execution_state",
        help="Persistent locks, batch markers and canonical journals",
    )
    p.add_argument("--enable-live", action="store_true")
    p.add_argument("--confirm-live", default="", help="Live mode requires exact value LIVE")
    p.add_argument("--ignore-acceptance", action="store_true", help="Deprecated; live mode refuses this bypass")
    return p.parse_args()


def _append_jsonl_fsync(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _empty_reconciliation() -> dict:
    return {"orders": [], "missing_order_ids": [], "requires_manual_reconciliation": False}


def _reconcile_phase(broker: QmtBroker, results: list[dict]) -> dict:
    ids = submitted_order_ids(results)
    reconciliation = (
        broker.reconcile_order_ids(ids, max_attempts=3, retry_delay_seconds=0.5)
        if ids
        else _empty_reconciliation()
    )
    if has_uncertain_submission(results):
        reconciliation["requires_manual_reconciliation"] = True
        reconciliation["uncertain_submit_exception"] = True
    reconciliation["submitted_order_ids"] = ids
    return reconciliation


def _quantity_deviations(plan, results: list[dict]) -> list[dict]:
    deviations: list[dict] = []
    for row in results:
        if str(row.get("status", "")) != "SUBMITTED":
            continue
        index = int(row.get("intent_index", -1))
        if index < 0 or index >= len(plan):
            deviations.append({"reason": "invalid_intent_index", "result": row})
            continue
        expected = int(plan[index].shares)
        observed = int(row.get("shares", 0) or 0)
        if expected != observed:
            deviations.append(
                {
                    "code": str(row.get("code", "")),
                    "expected_shares": expected,
                    "submitted_shares": observed,
                }
            )
    return deviations


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
        "recovery_command": "reconcile_qmt_execution.py",
    }
    try:
        recovery["observed_orders"] = broker.query_orders(cancelable_only=False)
        recovery["observed_cancelable_orders"] = broker.query_orders(cancelable_only=True)
        recovery["observed_trades"] = broker.query_trades()
    except Exception as exc:
        recovery["broker_query_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    _write_json(output / "startup_recovery.json", recovery)
    raise RuntimeError(
        "an active account execution lock already exists; run reconcile_qmt_execution.py "
        "and do not submit a new batch until recovery is explicitly acknowledged"
    )


def _runtime_report(
    *,
    start_of_day_equity: float,
    current_equity: float,
    target_codes: list[str],
    kill_switch: bool,
    binding: dict,
) -> dict:
    report = evaluate_runtime_risk(
        start_of_day_equity=start_of_day_equity,
        current_equity=current_equity,
        target_codes=target_codes,
        policy=RuntimeRiskPolicy(kill_switch=kill_switch),
    )
    report["binding"] = dict(binding)
    return report


def main() -> int:
    args = parse_args()
    if args.commission_rate < 0 or args.min_commission < 0:
        raise ValueError("commission settings must be non-negative")
    if args.lot_size <= 0:
        raise ValueError("lot size must be positive")
    if args.buy_buffer_bps < 0 or args.sell_buffer_bps < 0:
        raise ValueError("execution quote buffers must be non-negative")

    cost = CostConfig(
        commission_rate=float(args.commission_rate),
        min_commission=float(args.min_commission),
        lot_size=int(args.lot_size),
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    state_root = Path(args.state_dir).resolve()
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
    executable = broker.executable_prices(
        ticks,
        buy_buffer_bps=float(args.buy_buffer_bps),
        sell_buffer_bps=float(args.sell_buffer_bps),
    )
    prices = {code: float(v["last"] or v["buy"] or v["sell"]) for code, v in executable.items()}
    plan, effective_target_weights = build_target_weight_plan(
        target_weights,
        prices,
        positions,
        total_asset=total_asset,
        exposure=args.exposure,
        lot_size=cost.lot_size,
    )
    phases = split_order_plan(plan)
    buy_reserve_estimate = estimate_buy_cash_reserve(phases.buys, cost=cost)
    batch_id = execution_batch_id(
        signal_date=str(bundle.signal_date),
        strategy_sha256=bundle.strategy_sha256,
        target_weights=effective_target_weights,
        account_key=account_key,
    )
    batch_tag = batch_id[:12]

    binding = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "account_key": account_key,
        "batch_id": batch_id,
        "strategy_sha256": bundle.strategy_sha256,
        "target_file_sha256": bundle.target_file_sha256,
        "signal_date": str(bundle.signal_date),
        "expected_execution_session": str(bundle.expected_execution_session),
        "expires_after_session": str(bundle.expires_after_session),
    }
    snapshot = {
        "total_asset": total_asset,
        "cash": cash,
        "position_count": len(positions),
        "target_count": len(target_codes),
        "requested_target_weight_sum": float(sum(target_weights.values())),
        "effective_target_weight_sum": float(sum(effective_target_weights.values())),
        "sell_order_count": len(phases.sells),
        "buy_order_count": len(phases.buys),
        "estimated_buy_cash_required": buy_reserve_estimate["estimated_buy_cash_required"],
        "execution_cost": asdict(cost),
        "buy_buffer_bps": float(args.buy_buffer_bps),
        "sell_buffer_bps": float(args.sell_buffer_bps),
        **binding,
        "dry_run": not args.enable_live,
    }
    _write_json(out / "pretrade_snapshot.json", snapshot)
    pd.DataFrame(serialize_plan(plan)).to_csv(out / "order_plan.csv", index=False, encoding="utf-8-sig")
    _write_json(out / "buy_cash_reserve_estimate.json", buy_reserve_estimate)

    risk_report = validate_pretrade(
        plan,
        total_asset=total_asset,
        target_count=len(target_codes),
        target_weights=effective_target_weights,
    )
    _write_json(out / "pretrade_risk.json", risk_report)

    runtime_report = None
    if args.start_of_day_equity is not None:
        runtime_report = _runtime_report(
            start_of_day_equity=float(args.start_of_day_equity),
            current_equity=total_asset,
            target_codes=target_codes,
            kill_switch=bool(args.runtime_kill_switch),
            binding=binding,
        )
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

    journal = state_root / "journals" / f"{batch_id}.jsonl"
    if journal.exists():
        raise RuntimeError(f"orphan/replay execution journal already exists: {journal}")
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
            "sell_order_count": len(phases.sells),
            "buy_order_count": len(phases.buys),
            "effective_target_weight_sum": float(sum(effective_target_weights.values())),
            "journal_path": str(journal),
            "execution_cost": asdict(cost),
            "buy_buffer_bps": float(args.buy_buffer_bps),
            "sell_buffer_bps": float(args.sell_buffer_bps),
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

    try:
        _append_jsonl_fsync(
            journal,
            {
                "event": "EXECUTION_START",
                "batch_id": batch_id,
                "account_key": account_key,
                "batch_tag": batch_tag,
                "signal_date": str(bundle.signal_date),
                "expected_execution_session": str(bundle.expected_execution_session),
                "strategy_sha256": bundle.strategy_sha256,
                "planned_order_count": len(plan),
                "sell_order_count": len(phases.sells),
                "buy_order_count": len(phases.buys),
            },
        )
    except Exception:
        update_execution_batch(batch_marker, status="INCOMPLETE", details={"orders_submitted": 0})
        release_account_execution_lock(account_lock, batch_id=batch_id)
        raise
    _write_json(out / "execution_journal_pointer.json", {"batch_id": batch_id, "journal_path": str(journal)})

    def journal_event(event: dict) -> None:
        _append_jsonl_fsync(journal, {"batch_id": batch_id, "account_key": account_key, **event})

    def manual_stop(stage: str, details: dict) -> int:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "batch_id": batch_id,
            "account_key": account_key,
            "status": "MANUAL_RECONCILIATION",
            "stage": stage,
            **details,
        }
        update_execution_batch(batch_marker, status="MANUAL_RECONCILIATION", details=payload)
        try:
            _write_json(out / "execution_failure.json", payload)
        except Exception:
            pass
        try:
            journal_event({"event": "EXECUTION_STOP_MANUAL_RECONCILIATION", **payload})
        except Exception:
            pass
        return 4

    try:
        update_execution_batch(
            batch_marker,
            status="SELL_SUBMITTING",
            details={"planned_sell_orders": len(phases.sells)},
        )
        sell_results = broker.submit_plan(
            phases.sells,
            on_event=journal_event,
            cost=cost,
            batch_tag=batch_tag,
            phase="SELL",
        )
        _write_json(out / "sell_phase_results.json", {"orders": sell_results})
        pd.DataFrame(sell_results).to_csv(out / "sell_phase_results.csv", index=False, encoding="utf-8-sig")
        sell_reconciliation = _reconcile_phase(broker, sell_results)
        _write_json(out / "sell_phase_reconciliation.json", sell_reconciliation)
        journal_event({"event": "SELL_RECONCILIATION", **sell_reconciliation})

        if sell_reconciliation.get("requires_manual_reconciliation"):
            return manual_stop(
                "SELL_RECONCILIATION",
                {"reconciliation": sell_reconciliation, "results": sell_results},
            )
        sell_incomplete = incomplete_results(sell_results)
        if sell_incomplete:
            journal_event({"event": "EXECUTION_INCOMPLETE_PENDING_RELEASE", "stage": "SELL"})
            update_execution_batch(
                batch_marker,
                status="INCOMPLETE",
                details={"stage": "SELL", "results": sell_results},
            )
            release_account_execution_lock(account_lock, batch_id=batch_id)
            return 3

        after_sell_asset, after_sell_cash, after_sell_positions = broker.snapshot()
        sell_position_check = validate_sell_position_effect(
            positions,
            sell_results,
            after_sell_positions,
        )
        _write_json(out / "sell_phase_position_check.json", sell_position_check)
        if not sell_position_check["passed"]:
            return manual_stop("SELL_POSITION_CHECK", sell_position_check)

        after_sell_binding = {**binding, "phase": "AFTER_SELL"}
        after_sell_runtime = _runtime_report(
            start_of_day_equity=float(args.start_of_day_equity),
            current_equity=after_sell_asset,
            target_codes=target_codes,
            kill_switch=bool(args.runtime_kill_switch),
            binding=after_sell_binding,
        )
        _write_json(out / "after_sell_runtime_risk.json", after_sell_runtime)
        if not after_sell_runtime["passed"]:
            journal_event(
                {"event": "EXECUTION_INCOMPLETE_PENDING_RELEASE", "stage": "AFTER_SELL_RUNTIME_RISK"}
            )
            update_execution_batch(
                batch_marker,
                status="INCOMPLETE",
                details={"stage": "AFTER_SELL_RUNTIME_RISK", "runtime_risk": after_sell_runtime},
            )
            release_account_execution_lock(account_lock, batch_id=batch_id)
            return 3

        buy_risk = validate_pretrade(
            list(phases.buys),
            total_asset=after_sell_asset,
            target_count=len(target_codes),
            target_weights=effective_target_weights,
        )
        buy_reserve = estimate_buy_cash_reserve(phases.buys, cost=cost)
        buy_reserve["fresh_cash_before_buy"] = float(after_sell_cash)
        _write_json(out / "buy_phase_pretrade_risk.json", buy_risk)
        _write_json(out / "buy_phase_cash_reserve.json", buy_reserve)
        if not buy_risk["passed"]:
            journal_event({"event": "EXECUTION_INCOMPLETE_PENDING_RELEASE", "stage": "BUY_PRETRADE_RISK"})
            update_execution_batch(
                batch_marker,
                status="INCOMPLETE",
                details={"stage": "BUY_PRETRADE_RISK", "risk": buy_risk},
            )
            release_account_execution_lock(account_lock, batch_id=batch_id)
            return 3

        update_execution_batch(
            batch_marker,
            status="BUY_SUBMITTING",
            details={
                "planned_buy_orders": len(phases.buys),
                "fresh_cash_before_buy": float(after_sell_cash),
            },
        )
        buy_results = broker.submit_plan(
            phases.buys,
            on_event=journal_event,
            starting_cash=after_sell_cash,
            cost=cost,
            batch_tag=batch_tag,
            phase="BUY",
        )
        _write_json(out / "buy_phase_results.json", {"orders": buy_results})
        pd.DataFrame(buy_results).to_csv(out / "buy_phase_results.csv", index=False, encoding="utf-8-sig")
        buy_reconciliation = _reconcile_phase(broker, buy_results)
        _write_json(out / "buy_phase_reconciliation.json", buy_reconciliation)
        journal_event({"event": "BUY_RECONCILIATION", **buy_reconciliation})

        if buy_reconciliation.get("requires_manual_reconciliation"):
            return manual_stop(
                "BUY_RECONCILIATION",
                {"reconciliation": buy_reconciliation, "results": buy_results},
            )

        buy_incomplete = incomplete_results(buy_results)
        quantity_deviations = _quantity_deviations(phases.buys, buy_results)
        if quantity_deviations:
            _write_json(out / "buy_phase_quantity_deviations.json", {"deviations": quantity_deviations})
        all_results = sell_results + buy_results
        _write_json(out / "submitted_orders.json", {"orders": all_results})
        pd.DataFrame(all_results).to_csv(out / "submitted_orders.csv", index=False, encoding="utf-8-sig")

        final_asset, final_cash, final_positions = broker.snapshot()
        final_snapshot = {
            "total_asset": final_asset,
            "cash": final_cash,
            "position_count": len(final_positions),
            "batch_id": batch_id,
            "account_key": account_key,
        }
        _write_json(out / "final_account_snapshot.json", final_snapshot)

        if buy_incomplete or quantity_deviations:
            journal_event({"event": "EXECUTION_INCOMPLETE_PENDING_RELEASE", "stage": "BUY"})
            update_execution_batch(
                batch_marker,
                status="INCOMPLETE",
                details={
                    "stage": "BUY",
                    "incomplete_results": buy_incomplete,
                    "quantity_deviations": quantity_deviations,
                },
            )
            release_account_execution_lock(account_lock, batch_id=batch_id)
            return 3

        all_submitted_ids = sorted(
            set(submitted_order_ids(sell_results) + submitted_order_ids(buy_results))
        )
        journal_event(
            {
                "event": "EXECUTION_COMPLETE_PENDING_RELEASE",
                "submitted_order_ids": all_submitted_ids,
                "final_cash": final_cash,
            }
        )
        update_execution_batch(
            batch_marker,
            status="COMPLETED",
            details={"submitted_order_ids": all_submitted_ids, "final_snapshot": final_snapshot},
        )
        release_account_execution_lock(account_lock, batch_id=batch_id)
        return 0
    except Exception as exc:
        return manual_stop(
            "UNHANDLED_EXECUTION_EXCEPTION",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )


if __name__ == "__main__":
    raise SystemExit(main())
