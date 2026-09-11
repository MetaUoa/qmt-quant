from __future__ import annotations

import json

import pytest

from qmt_quant.execution_recovery import (
    assess_execution_recovery,
    load_execution_journal,
    recovery_report_sha256,
)


BATCH = "a" * 64
ACCOUNT = "b" * 64


def _batch(status: str = "MANUAL_RECONCILIATION") -> dict:
    return {"batch_id": BATCH, "account_key": ACCOUNT, "status": status}


def _lock() -> dict:
    return {"batch_id": BATCH, "account_key": ACCOUNT, "status": "ACTIVE"}


def _order(order_id: int, remark: str, *, remaining: int = 0) -> dict:
    return {
        "order_id": order_id,
        "code": "000001.SZ",
        "order_volume": 100,
        "traded_volume": 100 - remaining,
        "remaining_volume": remaining,
        "order_remark": remark,
    }


def test_reserved_batch_with_no_submit_attempts_is_safe_to_acknowledge() -> None:
    report = assess_execution_recovery(
        batch_marker=_batch("RESERVED"),
        account_lock=_lock(),
        journal_records=[{"batch_id": BATCH, "event": "EXECUTION_START"}],
        all_orders=[],
        cancelable_orders=[],
        trades=[],
    )
    assert report["safe_to_acknowledge"] is True
    assert report["outcome"] == "NO_KNOWN_SIDE_EFFECTS"


def test_known_terminal_submitted_order_is_safe_to_acknowledge() -> None:
    remark = f"qmtq:{BATCH[:12]}:S:0"
    journal = [
        {"batch_id": BATCH, "event": "SUBMIT_ATTEMPT", "order_remark": remark},
        {
            "batch_id": BATCH,
            "event": "RESULT",
            "order_remark": remark,
            "status": "SUBMITTED",
            "order_id": 7,
        },
    ]
    report = assess_execution_recovery(
        batch_marker=_batch(),
        account_lock=_lock(),
        journal_records=journal,
        all_orders=[_order(7, remark)],
        cancelable_orders=[],
        trades=[{"order_id": 7}],
    )
    assert report["safe_to_acknowledge"] is True
    assert report["known_order_ids"] == [7]
    assert report["observed_trade_order_ids"] == [7]


def test_cancelable_batch_order_blocks_recovery_acknowledgement() -> None:
    remark = f"qmtq:{BATCH[:12]}:B:0"
    journal = [
        {"batch_id": BATCH, "event": "SUBMIT_ATTEMPT", "order_remark": remark},
        {
            "batch_id": BATCH,
            "event": "RESULT",
            "order_remark": remark,
            "status": "SUBMITTED",
            "order_id": 9,
        },
    ]
    order = _order(9, remark, remaining=100)
    report = assess_execution_recovery(
        batch_marker=_batch(),
        account_lock=_lock(),
        journal_records=journal,
        all_orders=[order],
        cancelable_orders=[order],
        trades=[],
    )
    assert report["safe_to_acknowledge"] is False
    assert any("batch_orders_still_cancelable" in item for item in report["violations"])


def test_submit_exception_can_be_recovered_only_by_unique_tagged_broker_order() -> None:
    remark = f"qmtq:{BATCH[:12]}:S:0"
    journal = [
        {"batch_id": BATCH, "event": "SUBMIT_ATTEMPT", "order_remark": remark},
        {
            "batch_id": BATCH,
            "event": "RESULT",
            "order_remark": remark,
            "status": "SUBMIT_EXCEPTION",
            "order_id": 0,
        },
    ]
    report = assess_execution_recovery(
        batch_marker=_batch(),
        account_lock=_lock(),
        journal_records=journal,
        all_orders=[_order(11, remark)],
        cancelable_orders=[],
        trades=[],
    )
    assert report["safe_to_acknowledge"] is True
    assert report["recovered_order_ids"] == [11]


def test_unresolved_submit_attempt_remains_fail_closed() -> None:
    remark = f"qmtq:{BATCH[:12]}:S:0"
    report = assess_execution_recovery(
        batch_marker=_batch(),
        account_lock=_lock(),
        journal_records=[
            {"batch_id": BATCH, "event": "SUBMIT_ATTEMPT", "order_remark": remark}
        ],
        all_orders=[],
        cancelable_orders=[],
        trades=[],
    )
    assert report["safe_to_acknowledge"] is False
    assert any("unresolved_submit_side_effect" in item for item in report["violations"])


def test_unjournaled_batch_tagged_order_blocks_recovery() -> None:
    remark = f"qmtq:{BATCH[:12]}:B:3"
    report = assess_execution_recovery(
        batch_marker=_batch(),
        account_lock=_lock(),
        journal_records=[{"batch_id": BATCH, "event": "EXECUTION_START"}],
        all_orders=[_order(13, remark)],
        cancelable_orders=[],
        trades=[],
    )
    assert report["safe_to_acknowledge"] is False
    assert any("unjournaled_tagged_orders" in item for item in report["violations"])


def test_recovery_report_digest_ignores_timestamp_and_existing_digest() -> None:
    base = {"generated_at_utc": "one", "batch_id": BATCH, "safe_to_acknowledge": True}
    first = recovery_report_sha256(base)
    second = recovery_report_sha256(
        {**base, "generated_at_utc": "two", "report_sha256": "f" * 64}
    )
    assert first == second


def test_execution_journal_rejects_malformed_json(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps({"event": "OK"}) + "\n{broken\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="line 2"):
        load_execution_journal(path)
