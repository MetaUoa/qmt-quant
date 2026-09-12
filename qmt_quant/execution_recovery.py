from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence


RECOVERY_SCHEMA = "qmt-execution-recovery-v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_execution_journal(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        raise RuntimeError(f"execution journal is missing: {source}")
    records: list[dict] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"execution journal is invalid at line {line_number}: {source}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"execution journal line {line_number} is not an object")
        records.append(payload)
    return records


def recovery_report_sha256(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("generated_at_utc", None)
    payload.pop("report_sha256", None)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _strict_int(value: object, *, name: str, default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise RuntimeError(f"{name} must be a finite integer")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return int(default)
        try:
            return int(text)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be an integer") from exc
    raise RuntimeError(f"{name} must be an integer")


def _strict_float(value: object, *, name: str) -> float:
    if value is None or value == "" or isinstance(value, bool):
        raise RuntimeError(f"{name} must be numeric")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError as exc:
            raise RuntimeError(f"{name} must be numeric") from exc
    else:
        raise RuntimeError(f"{name} must be numeric")
    if not math.isfinite(number):
        raise RuntimeError(f"{name} must be finite")
    return number


def _positive_order_id(row: Mapping[str, object]) -> int:
    value = _strict_int(row.get("order_id"), name="order_id")
    return value if value > 0 else 0


def _order_map(rows: Sequence[Mapping[str, object]]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in rows:
        order_id = _positive_order_id(row)
        if order_id <= 0:
            continue
        if order_id in out:
            raise RuntimeError(f"duplicate broker order_id observed during recovery: {order_id}")
        out[order_id] = dict(row)
    return out


def _trade_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    order_id = _positive_order_id(row)
    if order_id <= 0:
        raise RuntimeError("trade recovery record is missing a positive order_id")
    traded_id = str(row.get("traded_id", "") or "").strip()
    code = str(row.get("code", "") or "").strip()
    volume = _strict_int(row.get("traded_volume"), name=f"trade.traded_volume:{order_id}")
    price = _strict_float(row.get("traded_price"), name=f"trade.traded_price:{order_id}")
    if volume <= 0 or price <= 0 or not code:
        raise RuntimeError(f"trade recovery record is invalid for order {order_id}")
    if traded_id:
        return ("TRADED_ID", order_id, traded_id)
    return ("FILL", order_id, code, volume, round(price, 8))


def _normalize_positions(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise RuntimeError("account snapshot positions must be a mapping")
    positions: dict[str, int] = {}
    for code, shares_value in value.items():
        shares = _strict_int(shares_value, name=f"account_snapshot.positions.{code}")
        if shares < 0:
            raise RuntimeError("account snapshot positions cannot be negative")
        if shares > 0:
            positions[str(code)] = shares
    return positions


def assess_execution_recovery(
    *,
    batch_marker: Mapping[str, object],
    account_lock: Mapping[str, object],
    journal_records: Sequence[Mapping[str, object]],
    all_orders: Sequence[Mapping[str, object]],
    cancelable_orders: Sequence[Mapping[str, object]],
    trades: Sequence[Mapping[str, object]],
    account_snapshot: Mapping[str, object] | None = None,
    cash_tolerance: float = 2.0,
) -> dict:
    batch_id = str(batch_marker.get("batch_id", ""))
    account_key = str(batch_marker.get("account_key", ""))
    violations: list[str] = []
    valid_batch_id = bool(_SHA256_RE.fullmatch(batch_id))
    if not valid_batch_id:
        violations.append("invalid_batch_id")
    if not _SHA256_RE.fullmatch(account_key):
        violations.append("invalid_account_key")
    if str(account_lock.get("batch_id", "")) != batch_id:
        violations.append("lock_batch_mismatch")
    if str(account_lock.get("account_key", "")) != account_key:
        violations.append("lock_account_mismatch")

    relevant = [
        dict(row)
        for row in journal_records
        if not row.get("batch_id") or str(row.get("batch_id")) == batch_id
    ]
    attempt_rows = [row for row in relevant if str(row.get("event", "")) == "SUBMIT_ATTEMPT"]
    result_rows = [row for row in relevant if str(row.get("event", "")) == "RESULT"]

    order_by_id = _order_map(all_orders)
    cancelable_ids = {_positive_order_id(row) for row in cancelable_orders}
    cancelable_ids.discard(0)
    trade_order_ids = {_positive_order_id(row) for row in trades}
    trade_order_ids.discard(0)

    batch_tag_prefix = f"qmtq:{batch_id[:12]}:" if valid_batch_id else ""
    attempt_remarks = {
        str(row.get("order_remark", ""))
        for row in attempt_rows
        if str(row.get("order_remark", ""))
    }
    results_by_remark: dict[str, list[dict]] = {}
    direct_submitted_ids: set[int] = set()
    for row in result_rows:
        remark = str(row.get("order_remark", ""))
        status = str(row.get("status", ""))
        if remark:
            results_by_remark.setdefault(remark, []).append(row)
        if status in {"SUBMITTED", "SUBMIT_EXCEPTION", "FAILED"}:
            if not remark:
                violations.append(f"submit_result_missing_recovery_tag:{status}")
            elif remark not in attempt_remarks:
                violations.append(f"submit_result_without_attempt:{remark}")
        if status == "SUBMITTED":
            order_id = _positive_order_id(row)
            if order_id <= 0:
                violations.append("submitted_result_missing_order_id")
            else:
                direct_submitted_ids.add(order_id)

    tagged_orders = [
        dict(row)
        for row in all_orders
        if batch_tag_prefix and str(row.get("order_remark", "")).startswith(batch_tag_prefix)
    ]
    tagged_by_remark: dict[str, list[dict]] = {}
    for row in tagged_orders:
        tagged_by_remark.setdefault(str(row.get("order_remark", "")), []).append(row)

    recovered_order_ids: set[int] = set()
    resolved_attempt_remarks: set[str] = set()
    for attempt in attempt_rows:
        remark = str(attempt.get("order_remark", ""))
        if not remark:
            violations.append("submit_attempt_missing_recovery_tag")
            continue
        if batch_tag_prefix and not remark.startswith(batch_tag_prefix):
            violations.append(f"submit_attempt_wrong_batch_tag:{remark}")
            continue
        if remark in resolved_attempt_remarks:
            violations.append(f"duplicate_submit_attempt:{remark}")
            continue
        resolved_attempt_remarks.add(remark)
        matching_results = results_by_remark.get(remark, [])
        if len(matching_results) > 1:
            violations.append(f"duplicate_submit_result:{remark}")
            continue
        result = matching_results[0] if matching_results else None
        result_status = str(result.get("status", "")) if result else ""
        if result is not None and result_status == "SUBMITTED":
            order_id = _positive_order_id(result)
            if order_id <= 0:
                violations.append(f"submitted_result_missing_order_id:{remark}")
            continue
        if result is not None and result_status not in {"SUBMIT_EXCEPTION", ""}:
            continue

        matches = tagged_by_remark.get(remark, [])
        if len(matches) == 1:
            order_id = _positive_order_id(matches[0])
            if order_id <= 0:
                violations.append(f"recovered_order_missing_id:{remark}")
            else:
                recovered_order_ids.add(order_id)
        elif len(matches) == 0:
            violations.append(f"unresolved_submit_side_effect:{remark}")
        else:
            violations.append(f"ambiguous_recovered_orders:{remark}")

    known_order_ids = direct_submitted_ids | recovered_order_ids
    tagged_order_ids = {_positive_order_id(row) for row in tagged_orders}
    tagged_order_ids.discard(0)
    extras = tagged_order_ids.difference(known_order_ids)
    if extras:
        violations.append("unjournaled_tagged_orders:" + ",".join(str(x) for x in sorted(extras)))

    missing_history = known_order_ids.difference(order_by_id)
    if missing_history:
        violations.append(
            "missing_broker_order_history:" + ",".join(str(x) for x in sorted(missing_history))
        )

    active = known_order_ids.intersection(cancelable_ids)
    if active:
        violations.append("batch_orders_still_cancelable:" + ",".join(str(x) for x in sorted(active)))

    callback_sink_attached = any(
        str(row.get("event", "")) == "BROKER_EVENT_SINK_ATTACHED" for row in relevant
    )
    callback_trade_rows = [
        row for row in relevant if str(row.get("event", "")) == "BROKER_TRADE"
    ]
    callback_trade_verified = not callback_sink_attached
    if callback_sink_attached:
        try:
            broker_counter = Counter(
                _trade_identity(row)
                for row in trades
                if _positive_order_id(row) in known_order_ids
            )
            callback_counter = Counter(
                _trade_identity(row)
                for row in callback_trade_rows
                if _positive_order_id(row) in known_order_ids
            )
            missing_callbacks = broker_counter - callback_counter
            orphan_callbacks = callback_counter - broker_counter
            if missing_callbacks:
                violations.append(f"missing_callback_trades:{sum(missing_callbacks.values())}")
            if orphan_callbacks:
                violations.append(f"orphan_callback_trades:{sum(orphan_callbacks.values())}")
            callback_trade_verified = not missing_callbacks and not orphan_callbacks
        except RuntimeError as exc:
            violations.append(f"callback_trade_reconstruction_error:{exc}")
            callback_trade_verified = False

    cash_checkpoint_rows = [
        row
        for row in relevant
        if str(row.get("event", "")) in {"SELL_CASH_RECONCILIATION", "BUY_CASH_RECONCILIATION"}
        and row.get("passed") is True
    ]
    position_checkpoint_rows = [
        row for row in relevant if str(row.get("event", "")) == "ACCOUNT_SNAPSHOT"
    ]
    cash_state_verified = account_snapshot is None or not known_order_ids
    position_state_verified = account_snapshot is None or not known_order_ids
    observed_cash: float | None = None
    expected_cash: float | None = None
    observed_positions: dict[str, int] | None = None
    expected_positions: dict[str, int] | None = None

    if account_snapshot is not None:
        observed_cash = _strict_float(account_snapshot.get("cash"), name="account_snapshot.cash")
        if observed_cash < 0:
            violations.append("account_snapshot_negative_cash")
        observed_positions = _normalize_positions(account_snapshot.get("positions", {}))
        if known_order_ids:
            if not cash_checkpoint_rows:
                violations.append("missing_cash_reconciliation_checkpoint")
                cash_state_verified = False
            else:
                latest_cash = cash_checkpoint_rows[-1]
                try:
                    expected_cash = _strict_float(
                        latest_cash.get("cash_after"), name="cash_checkpoint.cash_after"
                    )
                    tolerance = max(float(cash_tolerance), 0.0)
                    cash_state_verified = abs(observed_cash - expected_cash) <= tolerance
                    if not cash_state_verified:
                        violations.append("cash_state_drift_after_reconciliation")
                except RuntimeError as exc:
                    violations.append(f"cash_checkpoint_invalid:{exc}")
                    cash_state_verified = False

            if not position_checkpoint_rows:
                violations.append("missing_position_checkpoint")
                position_state_verified = False
            else:
                latest_position = position_checkpoint_rows[-1]
                try:
                    expected_positions = _normalize_positions(latest_position.get("positions", {}))
                    position_state_verified = observed_positions == expected_positions
                    if not position_state_verified:
                        violations.append("position_state_drift_after_checkpoint")
                except RuntimeError as exc:
                    violations.append(f"position_checkpoint_invalid:{exc}")
                    position_state_verified = False

    safe_to_acknowledge = not violations
    outcome = "BLOCKED"
    if safe_to_acknowledge:
        outcome = "TERMINAL_SIDE_EFFECTS" if known_order_ids else "NO_KNOWN_SIDE_EFFECTS"

    report: dict[str, object] = {
        "schema": RECOVERY_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "account_key": account_key,
        "batch_status": str(batch_marker.get("status", "")),
        "safe_to_acknowledge": safe_to_acknowledge,
        "outcome": outcome,
        "violations": violations,
        "journal_event_count": len(relevant),
        "submit_attempt_count": len(attempt_rows),
        "submitted_order_ids": sorted(direct_submitted_ids),
        "recovered_order_ids": sorted(recovered_order_ids),
        "known_order_ids": sorted(known_order_ids),
        "tagged_order_ids": sorted(tagged_order_ids),
        "active_order_ids": sorted(active),
        "observed_trade_order_ids": sorted(trade_order_ids.intersection(known_order_ids)),
        "callback_sink_attached": callback_sink_attached,
        "callback_trade_count": len(callback_trade_rows),
        "callback_trade_verified": callback_trade_verified,
        "cash_checkpoint_count": len(cash_checkpoint_rows),
        "cash_state_verified": cash_state_verified,
        "observed_cash": observed_cash,
        "expected_cash_checkpoint": expected_cash,
        "position_checkpoint_count": len(position_checkpoint_rows),
        "position_state_verified": position_state_verified,
        "observed_positions": observed_positions,
        "expected_positions_checkpoint": expected_positions,
    }
    report["report_sha256"] = recovery_report_sha256(report)
    return report
