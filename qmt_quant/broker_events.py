from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
from typing import Callable, Mapping


EventSink = Callable[[dict[str, object]], None]


class JsonlEventJournal:
    """Thread-safe fsync JSONL writer shared by executor and XtQuant callbacks."""

    def __init__(self, path: str | Path, *, context: Mapping[str, object] | None = None) -> None:
        self.path = Path(path)
        self.context = dict(context or {})
        self._lock = threading.Lock()

    def append(self, payload: Mapping[str, object]) -> None:
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **self.context,
            **dict(payload),
        }
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(obj: object, name: str) -> str:
    return str(getattr(obj, name, "") or "")


def _integer(obj: object, name: str) -> int | None:
    value = getattr(obj, name, None)
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _number(obj: object, name: str) -> float | None:
    value = getattr(obj, name, None)
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def callback_method_map(
    emit: EventSink,
    *,
    expected_account_id: str,
    on_disconnect: Callable[[], None] | None = None,
) -> dict[str, Callable[..., None]]:
    """Build methods for an XtQuantTraderCallback subclass without importing xtquant in CI."""

    def send(event: str, payload: dict[str, object]) -> None:
        emit(
            {
                "event": event,
                "broker_event_received_at_utc": _now_iso(),
                "expected_account_id": str(expected_account_id),
                **payload,
            }
        )

    def on_disconnected(self: object) -> None:
        if on_disconnect is not None:
            on_disconnect()
        send("BROKER_DISCONNECTED", {})

    def on_account_status(self: object, status: object) -> None:
        send(
            "BROKER_ACCOUNT_STATUS",
            {
                "account_id": _text(status, "account_id"),
                "account_type": _integer(status, "account_type"),
                "account_status": _integer(status, "status"),
            },
        )

    def on_stock_asset(self: object, asset: object) -> None:
        send(
            "BROKER_ASSET",
            {
                "account_id": _text(asset, "account_id"),
                "cash": _number(asset, "cash"),
                "total_asset": _number(asset, "total_asset"),
            },
        )

    def on_stock_order(self: object, order: object) -> None:
        send(
            "BROKER_ORDER",
            {
                "account_id": _text(order, "account_id"),
                "order_id": _integer(order, "order_id"),
                "order_sysid": _text(order, "order_sysid"),
                "code": _text(order, "stock_code"),
                "order_status": _integer(order, "order_status"),
                "order_volume": _integer(order, "order_volume"),
                "traded_volume": _integer(order, "traded_volume"),
                "order_remark": _text(order, "order_remark"),
                "strategy_name": _text(order, "strategy_name"),
            },
        )

    def on_stock_trade(self: object, trade: object) -> None:
        send(
            "BROKER_TRADE",
            {
                "account_id": _text(trade, "account_id"),
                "order_id": _integer(trade, "order_id"),
                "order_sysid": _text(trade, "order_sysid"),
                "traded_id": _text(trade, "traded_id"),
                "code": _text(trade, "stock_code"),
                "traded_volume": _integer(trade, "traded_volume"),
                "traded_price": _number(trade, "traded_price"),
                "order_remark": _text(trade, "order_remark"),
            },
        )

    def on_order_error(self: object, error: object) -> None:
        send(
            "BROKER_ORDER_ERROR",
            {
                "account_id": _text(error, "account_id"),
                "order_id": _integer(error, "order_id"),
                "error_id": _integer(error, "error_id"),
                "error_msg": _text(error, "error_msg"),
                "order_remark": _text(error, "order_remark"),
            },
        )

    def on_cancel_error(self: object, error: object) -> None:
        send(
            "BROKER_CANCEL_ERROR",
            {
                "account_id": _text(error, "account_id"),
                "order_id": _integer(error, "order_id"),
                "error_id": _integer(error, "error_id"),
                "error_msg": _text(error, "error_msg"),
            },
        )

    def on_order_stock_async_response(self: object, response: object) -> None:
        send(
            "BROKER_ORDER_ASYNC_RESPONSE",
            {
                "account_id": _text(response, "account_id"),
                "order_id": _integer(response, "order_id"),
                "seq": _integer(response, "seq"),
                "order_remark": _text(response, "order_remark"),
                "strategy_name": _text(response, "strategy_name"),
            },
        )

    return {
        "on_disconnected": on_disconnected,
        "on_account_status": on_account_status,
        "on_stock_asset": on_stock_asset,
        "on_stock_order": on_stock_order,
        "on_stock_trade": on_stock_trade,
        "on_order_error": on_order_error,
        "on_cancel_error": on_cancel_error,
        "on_order_stock_async_response": on_order_stock_async_response,
    }
