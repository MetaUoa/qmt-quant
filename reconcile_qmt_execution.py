from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from qmt_quant.execution_recovery import (
    RECOVERY_SCHEMA,
    assess_execution_recovery,
    load_execution_journal,
)
from qmt_quant.execution_state import (
    account_execution_key,
    account_lock_path,
    load_account_execution_lock,
    read_execution_state,
    release_account_execution_lock,
    update_execution_batch,
)
from qmt_quant.live_trader import QmtBroker


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile one locked qmt-quant execution batch without submitting orders"
    )
    parser.add_argument("--userdata", required=True, help="MiniQMT userdata_mini directory")
    parser.add_argument("--account", required=True)
    parser.add_argument("--account-type", default="STOCK")
    parser.add_argument("--session-id", type=int, default=26090201)
    parser.add_argument("--state-dir", default="output/execution_state")
    parser.add_argument("--output", default="output/execution_recovery/recovery_report.json")
    parser.add_argument("--cash-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--acknowledge-batch",
        default="",
        help="Exact locked batch SHA256. Releases the account lock only when recovery is proven terminal.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.cash_tolerance < 0:
        raise ValueError("--cash-tolerance must be non-negative")
    state_root = Path(args.state_dir).resolve()
    output = Path(args.output)
    account_key = account_execution_key(account_id=args.account, account_type=args.account_type)
    lock = load_account_execution_lock(state_root, account_key=account_key)
    if lock is None:
        payload = {
            "schema": RECOVERY_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "account_key": account_key,
            "status": "NO_ACTIVE_ACCOUNT_LOCK",
            "safe_to_acknowledge": False,
        }
        _write_json(output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    batch_id = str(lock.get("batch_id", ""))
    if not _SHA256_RE.fullmatch(batch_id):
        raise RuntimeError("active account lock contains an invalid batch id")
    batch_path = state_root / "execution_batches" / f"{batch_id}.json"
    batch = read_execution_state(batch_path)
    if str(batch.get("account_key", "")) != account_key:
        raise RuntimeError("batch marker account identity does not match active lock")
    journal_value = str(batch.get("journal_path", ""))
    if not journal_value:
        raise RuntimeError(
            "batch marker does not contain canonical journal_path; conservative automatic recovery is unavailable"
        )
    journal_path = Path(journal_value)
    if not journal_path.is_absolute():
        raise RuntimeError("canonical journal_path must be absolute")
    journal = load_execution_journal(journal_path)

    broker = QmtBroker(args.userdata, args.account, args.session_id, args.account_type)
    broker.connect(max_attempts=3, retry_delay_seconds=1.0)
    all_orders = broker.query_orders(cancelable_only=False)
    cancelable_orders = broker.query_orders(cancelable_only=True)
    trades = broker.query_trades()
    total_asset, cash, positions = broker.snapshot()
    report = assess_execution_recovery(
        batch_marker=batch,
        account_lock=lock,
        journal_records=journal,
        all_orders=all_orders,
        cancelable_orders=cancelable_orders,
        trades=trades,
        account_snapshot={
            "total_asset": float(total_asset),
            "cash": float(cash),
            "positions": dict(positions),
        },
        cash_tolerance=float(args.cash_tolerance),
    )
    report["batch_marker_path"] = str(batch_path)
    report["journal_path"] = str(journal_path)
    report["account_lock_path"] = str(account_lock_path(state_root, account_key=account_key))

    acknowledge = str(args.acknowledge_batch).strip().lower()
    if acknowledge:
        if acknowledge != batch_id:
            report["acknowledgement"] = "REFUSED_BATCH_ID_MISMATCH"
            _write_json(output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 4
        if report.get("safe_to_acknowledge") is not True:
            report["acknowledgement"] = "REFUSED_RECOVERY_NOT_TERMINAL"
            _write_json(output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 4
        update_execution_batch(
            batch_path,
            status="RECOVERED_RELEASED",
            details={
                "recovery_report_sha256": report["report_sha256"],
                "recovery_outcome": report["outcome"],
                "known_order_ids": report["known_order_ids"],
                "callback_trade_verified": report["callback_trade_verified"],
                "cash_state_verified": report["cash_state_verified"],
                "position_state_verified": report["position_state_verified"],
            },
        )
        release_account_execution_lock(
            account_lock_path(state_root, account_key=account_key),
            batch_id=batch_id,
        )
        report["acknowledgement"] = "ACCEPTED_LOCK_RELEASED"
        _write_json(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    report["acknowledgement"] = "NOT_REQUESTED"
    _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if report.get("safe_to_acknowledge") is True else 4


if __name__ == "__main__":
    raise SystemExit(main())
