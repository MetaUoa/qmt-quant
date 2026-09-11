from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest

from qmt_quant.freshness import FreshnessPolicy
from qmt_quant.live_trader import BrokerStateUnknown, OrderInstruction, QmtBroker


def _bare_broker() -> QmtBroker:
    broker = object.__new__(QmtBroker)
    broker.userdata_path = "unused"
    broker.account_id = "acct"
    broker.session_id = 1
    broker.account_type = "STOCK"
    broker.trader = object()
    broker.account = object()
    broker._callback = None
    broker._event_sink = None
    broker._event_buffer = []
    broker._event_sink_error = None
    broker._connection_lost = False
    broker._connected_at_utc = None
    broker._last_broker_event_at_utc = None
    return broker


def _install_xtconstant(monkeypatch: pytest.MonkeyPatch) -> None:
    xtquant = ModuleType("xtquant")
    xtquant.xtconstant = SimpleNamespace(STOCK_BUY=23, STOCK_SELL=24, FIX_PRICE=11)
    monkeypatch.setitem(sys.modules, "xtquant", xtquant)


def test_disconnect_callback_state_blocks_all_subsequent_broker_operations() -> None:
    broker = _bare_broker()
    broker._mark_disconnected()
    assert broker.broker_health()["connection_lost"] is True
    with pytest.raises(BrokerStateUnknown, match="disconnect callback"):
        broker._require_connection_healthy()


def test_callback_sink_failure_poison_state_blocks_new_orders() -> None:
    broker = _bare_broker()

    def broken_sink(_event: dict[str, object]) -> None:
        raise OSError("disk failed")

    broker.attach_event_sink(broken_sink)
    broker._record_broker_event({"event": "BROKER_ORDER"})
    assert broker.broker_health()["event_sink_failed"] is True
    with pytest.raises(BrokerStateUnknown, match="journal failed"):
        broker._require_connection_healthy()


def test_buffered_callback_failure_during_sink_attach_fails_closed() -> None:
    broker = _bare_broker()
    broker._record_broker_event({"event": "BROKER_CONNECTED"})

    def broken_sink(_event: dict[str, object]) -> None:
        raise OSError("read only filesystem")

    with pytest.raises(BrokerStateUnknown, match="flush buffered"):
        broker.attach_event_sink(broken_sink)
    assert broker.broker_health()["event_sink_failed"] is True


def test_submit_plan_rejects_stale_quotes_before_order_stock(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_xtconstant(monkeypatch)

    class Trader:
        def __init__(self) -> None:
            self.order_calls = 0

        def order_stock(self, *_args: object, **_kwargs: object) -> int:
            self.order_calls += 1
            return 7

    trader = Trader()
    broker = _bare_broker()
    broker.trader = trader
    broker.full_tick = lambda _codes: {
        "000001.SZ": {
            "time": 1,
            "lastPrice": 10.0,
            "askPrice": [10.0],
            "bidPrice": [10.0],
            "lastClose": 9.9,
        }
    }
    with pytest.raises(BrokerStateUnknown, match="freshness gate failed"):
        broker.submit_plan(
            [OrderInstruction("000001.SZ", "SELL", 100, 10.0, "reduce")],
            freshness_policy=FreshnessPolicy(quote_max_age_seconds=5),
            require_fresh_quotes=True,
            batch_tag="a" * 12,
            phase="SELL",
        )
    assert trader.order_calls == 0


def test_callback_buffer_is_flushed_in_original_order() -> None:
    broker = _bare_broker()
    broker._record_broker_event({"event": "ONE"})
    broker._record_broker_event({"event": "TWO"})
    observed: list[str] = []
    broker.attach_event_sink(lambda event: observed.append(str(event["event"])))
    assert observed == ["ONE", "TWO"]
    assert broker.broker_health()["buffered_event_count"] == 0
