from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading

from qmt_quant.broker_events import JsonlEventJournal, callback_method_map


def test_callback_methods_normalize_order_trade_error_and_disconnect() -> None:
    events: list[dict[str, object]] = []
    disconnected = []
    methods = callback_method_map(
        events.append,
        expected_account_id="acct-1",
        on_disconnect=lambda: disconnected.append(True),
    )
    callback = object()

    methods["on_stock_order"](
        callback,
        SimpleNamespace(
            account_id="acct-1",
            order_id=7,
            order_sysid="sys-7",
            stock_code="000001.SZ",
            order_status=56,
            order_volume=100,
            traded_volume=100,
            order_remark="qmtq:abc:S:0",
            strategy_name="qmt_quant_v7",
        ),
    )
    methods["on_stock_trade"](
        callback,
        SimpleNamespace(
            account_id="acct-1",
            order_id=7,
            order_sysid="sys-7",
            traded_id="trade-1",
            stock_code="000001.SZ",
            traded_volume=100,
            traded_price=10.01,
            order_remark="qmtq:abc:S:0",
        ),
    )
    methods["on_order_error"](
        callback,
        SimpleNamespace(
            account_id="acct-1",
            order_id=8,
            error_id=42,
            error_msg="rejected",
            order_remark="qmtq:abc:B:0",
        ),
    )
    methods["on_disconnected"](callback)

    assert disconnected == [True]
    assert [row["event"] for row in events] == [
        "BROKER_ORDER",
        "BROKER_TRADE",
        "BROKER_ORDER_ERROR",
        "BROKER_DISCONNECTED",
    ]
    assert events[0]["order_id"] == 7
    assert events[1]["traded_price"] == 10.01
    assert events[2]["error_id"] == 42
    assert all(row["expected_account_id"] == "acct-1" for row in events)
    assert all("broker_event_received_at_utc" in row for row in events)


def test_jsonl_event_journal_serializes_concurrent_writers(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    writer = JsonlEventJournal(path, context={"batch_id": "a" * 64})

    def write_block(worker: int) -> None:
        for index in range(25):
            writer.append({"event": "CALLBACK", "worker": worker, "index": index})

    threads = [threading.Thread(target=write_block, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 100
    assert all(record["batch_id"] == "a" * 64 for record in records)
    assert all(record["event"] == "CALLBACK" for record in records)
    assert all("recorded_at_utc" in record for record in records)
