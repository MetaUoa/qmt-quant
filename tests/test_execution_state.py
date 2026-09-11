from __future__ import annotations

import json

import pytest

from qmt_quant.execution_state import (
    account_execution_key,
    batch_status_requires_recovery,
    execution_batch_id,
    load_account_execution_lock,
    release_account_execution_lock,
    reserve_account_execution_lock,
    reserve_execution_batch,
    update_execution_batch,
)


def test_execution_batch_id_is_order_independent_for_target_mapping() -> None:
    a = execution_batch_id(
        signal_date="2026-09-12",
        strategy_sha256="a" * 64,
        target_weights={"000001.SZ": 0.2, "000002.SZ": 0.1},
    )
    b = execution_batch_id(
        signal_date="2026-09-12",
        strategy_sha256="a" * 64,
        target_weights={"000002.SZ": 0.1, "000001.SZ": 0.2},
    )
    assert a == b


def test_account_key_and_batch_identity_are_account_scoped() -> None:
    account_a = account_execution_key(account_id="A001", account_type="STOCK")
    account_b = account_execution_key(account_id="A002", account_type="STOCK")
    assert account_a != account_b
    common = {
        "signal_date": "2026-09-12",
        "strategy_sha256": "a" * 64,
        "target_weights": {"000001.SZ": 0.2},
    }
    assert execution_batch_id(**common, account_key=account_a) != execution_batch_id(
        **common, account_key=account_b
    )


def test_execution_batch_reservation_is_atomic_and_persistent(tmp_path) -> None:
    batch_id = "b" * 64
    marker = reserve_execution_batch(
        tmp_path,
        batch_id=batch_id,
        metadata={"signal_date": "2026-09-12"},
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "RESERVED"
    with pytest.raises(RuntimeError, match="already exists"):
        reserve_execution_batch(
            tmp_path,
            batch_id=batch_id,
            metadata={"signal_date": "2026-09-12"},
        )


def test_execution_batch_status_update_keeps_marker_reserved_for_replay_guard(tmp_path) -> None:
    batch_id = "c" * 64
    marker = reserve_execution_batch(tmp_path, batch_id=batch_id, metadata={})
    update_execution_batch(marker, status="COMPLETED", details={"submitted_order_ids": [7]})
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETED"
    assert payload["details"]["submitted_order_ids"] == [7]
    with pytest.raises(RuntimeError, match="already exists"):
        reserve_execution_batch(tmp_path, batch_id=batch_id, metadata={})


def test_account_lock_blocks_different_batches_until_explicit_release(tmp_path) -> None:
    account_key = account_execution_key(account_id="A001", account_type="STOCK")
    lock = reserve_account_execution_lock(
        tmp_path,
        account_key=account_key,
        batch_id="a" * 64,
        metadata={"account_type": "STOCK"},
    )
    loaded = load_account_execution_lock(tmp_path, account_key=account_key)
    assert loaded is not None
    assert loaded["batch_id"] == "a" * 64
    with pytest.raises(RuntimeError, match="account execution lock already exists"):
        reserve_account_execution_lock(
            tmp_path,
            account_key=account_key,
            batch_id="b" * 64,
        )
    release_account_execution_lock(lock, batch_id="a" * 64)
    assert load_account_execution_lock(tmp_path, account_key=account_key) is None


def test_account_lock_cannot_be_released_by_another_batch(tmp_path) -> None:
    account_key = account_execution_key(account_id="A001", account_type="STOCK")
    lock = reserve_account_execution_lock(
        tmp_path,
        account_key=account_key,
        batch_id="a" * 64,
    )
    with pytest.raises(RuntimeError, match="another batch"):
        release_account_execution_lock(lock, batch_id="b" * 64)
    assert lock.exists()


def test_nonterminal_batch_status_requires_recovery() -> None:
    assert batch_status_requires_recovery("RESERVED") is True
    assert batch_status_requires_recovery("MANUAL_RECONCILIATION") is True
    assert batch_status_requires_recovery("COMPLETED") is False
    assert batch_status_requires_recovery("INCOMPLETE") is False
