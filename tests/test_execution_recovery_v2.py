from __future__ import annotations

from qmt_quant.execution_recovery import assess_execution_recovery


BATCH = "a" * 64
ACCOUNT = "b" * 64
REMARK = f"qmtq:{BATCH[:12]}:B:0"


def _journal(*, include_position_checkpoint: bool = True, callback_price: float = 10.0) -> list[dict]:
    rows = [
        {"batch_id": BATCH, "event": "EXECUTION_START"},
        {"batch_id": BATCH, "event": "BROKER_EVENT_SINK_ATTACHED"},
        {
            "batch_id": BATCH,
            "event": "SUBMIT_ATTEMPT",
            "order_remark": REMARK,
            "side": "BUY",
            "code": "000001.SZ",
            "shares": 100,
        },
        {
            "batch_id": BATCH,
            "event": "RESULT",
            "order_remark": REMARK,
            "side": "BUY",
            "code": "000001.SZ",
            "shares": 100,
            "status": "SUBMITTED",
            "order_id": 7,
        },
        {
            "batch_id": BATCH,
            "event": "BROKER_TRADE",
            "order_id": 7,
            "traded_id": "T7",
            "code": "000001.SZ",
            "traded_volume": 100,
            "traded_price": callback_price,
        },
        {
            "batch_id": BATCH,
            "event": "BUY_CASH_RECONCILIATION",
            "passed": True,
            "cash_after": 8994.9,
        },
    ]
    if include_position_checkpoint:
        rows.append(
            {
                "batch_id": BATCH,
                "event": "ACCOUNT_SNAPSHOT",
                "cash": 8994.9,
                "positions": {"000001.SZ": 100},
            }
        )
    return rows


def _order() -> dict:
    return {
        "order_id": 7,
        "code": "000001.SZ",
        "order_volume": 100,
        "traded_volume": 100,
        "remaining_volume": 0,
        "order_remark": REMARK,
    }


def _trade() -> dict:
    return {
        "order_id": 7,
        "traded_id": "T7",
        "code": "000001.SZ",
        "traded_volume": 100,
        "traded_price": 10.0,
        "order_remark": REMARK,
    }


def _assess(journal: list[dict], *, cash: float = 8994.9) -> dict:
    return assess_execution_recovery(
        batch_marker={"batch_id": BATCH, "account_key": ACCOUNT, "status": "MANUAL_RECONCILIATION"},
        account_lock={"batch_id": BATCH, "account_key": ACCOUNT},
        journal_records=journal,
        all_orders=[_order()],
        cancelable_orders=[],
        trades=[_trade()],
        account_snapshot={"cash": cash, "positions": {"000001.SZ": 100}},
        cash_tolerance=0.01,
    )


def test_recovery_v2_verifies_callback_cash_and_position_checkpoints() -> None:
    report = _assess(_journal())
    assert report["safe_to_acknowledge"] is True
    assert report["callback_trade_verified"] is True
    assert report["cash_state_verified"] is True
    assert report["position_state_verified"] is True


def test_recovery_v2_refuses_side_effect_release_without_position_checkpoint() -> None:
    report = _assess(_journal(include_position_checkpoint=False))
    assert report["safe_to_acknowledge"] is False
    assert "missing_position_checkpoint" in report["violations"]


def test_recovery_v2_refuses_callback_trade_mismatch() -> None:
    rows = _journal(callback_price=11.0)
    rows[4]["traded_id"] = ""
    broker = _trade()
    broker["traded_id"] = ""
    report = assess_execution_recovery(
        batch_marker={"batch_id": BATCH, "account_key": ACCOUNT, "status": "MANUAL_RECONCILIATION"},
        account_lock={"batch_id": BATCH, "account_key": ACCOUNT},
        journal_records=rows,
        all_orders=[_order()],
        cancelable_orders=[],
        trades=[broker],
        account_snapshot={"cash": 8994.9, "positions": {"000001.SZ": 100}},
        cash_tolerance=0.01,
    )
    assert report["safe_to_acknowledge"] is False
    assert any("callback_trades" in item for item in report["violations"])


def test_recovery_v2_refuses_cash_drift_after_checkpoint() -> None:
    report = _assess(_journal(), cash=9000.0)
    assert report["safe_to_acknowledge"] is False
    assert "cash_state_drift_after_reconciliation" in report["violations"]
