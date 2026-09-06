from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

from qmt_quant.live_trader import OrderInstruction, QmtBroker


def _bare_broker():
    broker = object.__new__(QmtBroker)
    broker.userdata_path = "unused"
    broker.account_id = "unused"
    broker.session_id = 1
    broker.account_type = "STOCK"
    broker.trader = None
    broker.account = None
    return broker


def test_connect_retries_with_fresh_trader_instances():
    attempts = []

    class Trader:
        def __init__(self, _path, _session):
            self.index = len(attempts)
            attempts.append(self)
            self.stopped = False

        def start(self):
            return None

        def connect(self):
            return 1 if self.index == 0 else 0

        def subscribe(self, _account):
            return 0

        def stop(self):
            self.stopped = True

    broker = _bare_broker()
    broker._XtQuantTrader = Trader
    broker._StockAccount = lambda account_id, account_type: (account_id, account_type)
    broker.connect(max_attempts=2, retry_delay_seconds=0)
    assert len(attempts) == 2
    assert attempts[0].stopped is True
    assert broker.trader is attempts[1]


def test_submit_exception_is_journaled_and_stops_batch(monkeypatch):
    xtquant = ModuleType("xtquant")
    xtquant.xtconstant = SimpleNamespace(STOCK_BUY=23, STOCK_SELL=24, FIX_PRICE=11)
    monkeypatch.setitem(sys.modules, "xtquant", xtquant)

    class Trader:
        def order_stock(self, *_args, **_kwargs):
            raise TimeoutError("uncertain remote outcome")

    broker = _bare_broker()
    broker.trader = Trader()
    broker.account = object()
    broker.full_tick = lambda _codes: {
        "000001.SZ": {
            "lastPrice": 10.0,
            "askPrice": [10.0],
            "bidPrice": [10.0],
            "lastClose": 9.9,
        }
    }
    events = []
    plan = [
        OrderInstruction("000001.SZ", "SELL", 100, 10.0, "first"),
        OrderInstruction("000002.SZ", "SELL", 100, 10.0, "must_not_run"),
    ]
    results = broker.submit_plan(plan, on_event=events.append)
    assert len(results) == 1
    assert results[0]["status"] == "SUBMIT_EXCEPTION"
    assert [row["event"] for row in events] == ["INTENT", "SUBMIT_ATTEMPT", "RESULT"]


def test_reconcile_flags_partial_fill_for_manual_action():
    broker = _bare_broker()
    broker.trader = SimpleNamespace(
        query_stock_orders=lambda _account, _cancelable: [
            SimpleNamespace(
                order_id=7,
                stock_code="000001.SZ",
                order_volume=1000,
                traded_volume=400,
                order_status=50,
            )
        ]
    )
    broker.account = object()
    result = broker.reconcile_order_ids([7], max_attempts=1, retry_delay_seconds=0)
    assert result["orders"][0]["remaining_volume"] == 600
    assert result["requires_manual_reconciliation"] is True
