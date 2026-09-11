from __future__ import annotations

import json

import pytest

from qmt_quant.execution_state import (
    execution_batch_id,
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
