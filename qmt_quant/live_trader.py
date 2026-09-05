from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable, Iterable


@dataclass(frozen=True)
class PositionSnapshot:
    code: str
    volume: int
    available: int
    market_value: float = 0.0


@dataclass(frozen=True)
class OrderInstruction:
    code: str
    side: str
    shares: int
    reference_price: float
    reason: str


def build_equal_weight_plan(
    target_codes: Iterable[str],
    prices: dict[str, float],
    positions: dict[str, PositionSnapshot],
    *,
    total_asset: float,
    exposure: float = 1.0,
    lot_size: int = 100,
) -> list[OrderInstruction]:
    targets = [c for c in dict.fromkeys(target_codes) if c in prices and float(prices[c]) > 0]
    exposure = min(max(float(exposure), 0.0), 1.0)
    target_value = float(total_asset) * exposure / len(targets) if targets else 0.0
    desired: dict[str, int] = {}
    for code in targets:
        px = float(prices[code])
        desired[code] = max(int(target_value // (px * lot_size)), 0) * lot_size

    orders: list[OrderInstruction] = []
    # Sells first. Never plan more than today's available volume (T+1 safe).
    for code, pos in sorted(positions.items()):
        current = int(pos.volume)
        target = int(desired.get(code, 0))
        qty = min(max(current - target, 0), max(int(pos.available), 0))
        qty = qty // lot_size * lot_size
        px = float(prices.get(code, 0.0))
        if qty >= lot_size and px > 0:
            orders.append(OrderInstruction(code, "SELL", qty, px, "rebalance_reduce"))

    for code in targets:
        current = int(positions.get(code, PositionSnapshot(code, 0, 0)).volume)
        target = int(desired.get(code, 0))
        qty = max(target - current, 0)
        qty = qty // lot_size * lot_size
        if qty >= lot_size:
            orders.append(OrderInstruction(code, "BUY", qty, float(prices[code]), "rebalance_increase"))
    return orders


def serialize_plan(plan: list[OrderInstruction]) -> list[dict]:
    return [asdict(item) for item in plan]


class QmtBroker:
    """Thin MiniQMT execution adapter.

    Live mutation is intentionally isolated here. Callers should default to dry-run and
    require a separate explicit arming flag before invoking submit_plan().
    """

    def __init__(self, userdata_path: str, account_id: str, session_id: int, account_type: str = "STOCK") -> None:
        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
        except ImportError as exc:
            raise RuntimeError("xtquant trading modules are not available in this Python environment") from exc
        self._XtQuantTrader = XtQuantTrader
        self._StockAccount = StockAccount
        self.userdata_path = userdata_path
        self.account_id = account_id
        self.session_id = int(session_id)
        self.account_type = account_type
        self.trader = None
        self.account = None

    def connect(self, *, max_attempts: int = 3, retry_delay_seconds: float = 1.0) -> None:
        attempts = max(int(max_attempts), 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            trader = self._XtQuantTrader(self.userdata_path, self.session_id)
            account = self._StockAccount(self.account_id, self.account_type)
            try:
                trader.start()
                rc = trader.connect()
                if rc != 0:
                    raise RuntimeError(f"MiniQMT connect failed: {rc}")
                sub = trader.subscribe(account)
                if sub not in (0, True):
                    raise RuntimeError(f"MiniQMT account subscribe failed: {sub}")
                self.trader = trader
                self.account = account
                return
            except Exception as exc:
                last_error = exc
                stop = getattr(trader, "stop", None)
                if callable(stop):
                    try:
                        stop()
                    except Exception:
                        pass
                if attempt < attempts:
                    time.sleep(max(float(retry_delay_seconds), 0.0))
        raise RuntimeError(f"MiniQMT connect failed after {attempts} attempts") from last_error

    def snapshot(self) -> tuple[float, float, dict[str, PositionSnapshot]]:
        if self.trader is None or self.account is None:
            raise RuntimeError("Broker is not connected")
        asset = self.trader.query_stock_asset(self.account)
        if asset is None:
            raise RuntimeError("query_stock_asset returned None")
        positions = self.trader.query_stock_positions(self.account) or []
        mapped: dict[str, PositionSnapshot] = {}
        for p in positions:
            code = str(getattr(p, "stock_code"))
            mapped[code] = PositionSnapshot(
                code=code,
                volume=int(getattr(p, "volume", 0)),
                available=int(getattr(p, "can_use_volume", 0)),
                market_value=float(getattr(p, "market_value", 0.0)),
            )
        total_asset = float(getattr(asset, "total_asset", 0.0))
        if total_asset <= 0:
            total_asset = float(getattr(asset, "balance", 0.0) or getattr(asset, "cash", 0.0)) + sum(
                p.market_value for p in mapped.values()
            )
        cash = float(getattr(asset, "cash", 0.0))
        return total_asset, cash, mapped

    @staticmethod
    def full_tick(codes: Iterable[str]) -> dict:
        try:
            from xtquant import xtdata
        except ImportError as exc:
            raise RuntimeError("xtquant.xtdata is unavailable") from exc
        code_list = list(dict.fromkeys(codes))
        return xtdata.get_full_tick(code_list) or {}

    @staticmethod
    def executable_prices(ticks: dict, *, buy_buffer_bps: float = 8.0, sell_buffer_bps: float = 8.0) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for code, tick in ticks.items():
            last = float(tick.get("lastPrice") or 0.0)
            ask = tick.get("askPrice") or []
            bid = tick.get("bidPrice") or []
            buy_tradable = bool(ask and float(ask[0] or 0) > 0)
            sell_tradable = bool(bid and float(bid[0] or 0) > 0)
            buy_base = float(ask[0]) if buy_tradable else last
            sell_base = float(bid[0]) if sell_tradable else last
            if last <= 0:
                continue
            out[code] = {
                "last": last,
                "buy": buy_base * (1.0 + buy_buffer_bps / 10_000.0),
                "sell": sell_base * (1.0 - sell_buffer_bps / 10_000.0),
                "buy_tradable": buy_tradable,
                "sell_tradable": sell_tradable,
                "lastClose": float(tick.get("lastClose") or 0.0),
            }
        return out

    def query_orders(self, *, cancelable_only: bool = False) -> list[dict]:
        if self.trader is None or self.account is None:
            raise RuntimeError("Broker is not connected")
        rows = self.trader.query_stock_orders(self.account, bool(cancelable_only)) or []
        out: list[dict] = []
        for row in rows:
            order_id = int(getattr(row, "order_id", 0) or 0)
            order_volume = int(getattr(row, "order_volume", 0) or 0)
            traded_volume = int(getattr(row, "traded_volume", 0) or 0)
            out.append(
                {
                    "order_id": order_id,
                    "code": str(getattr(row, "stock_code", "")),
                    "order_volume": order_volume,
                    "traded_volume": traded_volume,
                    "remaining_volume": max(order_volume - traded_volume, 0),
                    "order_status": int(getattr(row, "order_status", -1) or -1),
                }
            )
        return out

    def reconcile_order_ids(
        self,
        order_ids: Iterable[int],
        *,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
    ) -> dict:
        wanted = {int(x) for x in order_ids if int(x) > 0}
        found: dict[int, dict] = {}
        for attempt in range(max(int(max_attempts), 1)):
            for row in self.query_orders(cancelable_only=False):
                if int(row["order_id"]) in wanted:
                    found[int(row["order_id"])] = row
            if wanted.issubset(found):
                break
            if attempt + 1 < max(int(max_attempts), 1):
                time.sleep(max(float(retry_delay_seconds), 0.0))
        return {
            "orders": [found[key] for key in sorted(found)],
            "missing_order_ids": sorted(wanted.difference(found)),
            "requires_manual_reconciliation": bool(
                wanted.difference(found)
                or any(int(row.get("remaining_volume", 0)) > 0 for row in found.values())
            ),
        }

    def submit_plan(
        self,
        plan: list[OrderInstruction],
        *,
        strategy_name: str = "qmt_quant_v7",
        on_event: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        if self.trader is None or self.account is None:
            raise RuntimeError("Broker is not connected")
        from xtquant import xtconstant

        def emit(payload: dict) -> None:
            if on_event is not None:
                on_event(dict(payload))

        codes = [x.code for x in plan]
        tick_prices = self.executable_prices(self.full_tick(codes))
        results: list[dict] = []
        for item in plan:
            emit(
                {
                    "event": "INTENT",
                    "code": item.code,
                    "side": item.side,
                    "shares": int(item.shares),
                    "reference_price": float(item.reference_price),
                    "reason": item.reason,
                }
            )
            p = tick_prices.get(item.code)
            if p is None:
                result = {"code": item.code, "side": item.side, "status": "SKIP_NO_TICK"}
                results.append(result)
                emit({"event": "RESULT", **result})
                continue
            if item.side == "BUY" and not p.get("buy_tradable", False):
                result = {"code": item.code, "side": item.side, "status": "SKIP_NO_ASK_LIMIT_OR_QUOTE"}
                results.append(result)
                emit({"event": "RESULT", **result})
                continue
            if item.side == "SELL" and not p.get("sell_tradable", False):
                result = {"code": item.code, "side": item.side, "status": "SKIP_NO_BID_LIMIT_OR_QUOTE"}
                results.append(result)
                emit({"event": "RESULT", **result})
                continue
            order_type = xtconstant.STOCK_BUY if item.side == "BUY" else xtconstant.STOCK_SELL
            price = p["buy"] if item.side == "BUY" else p["sell"]
            shares = int(item.shares)
            if item.side == "BUY":
                asset_now = self.trader.query_stock_asset(self.account)
                cash_now = float(getattr(asset_now, "cash", 0.0)) if asset_now is not None else 0.0
                affordable = int(cash_now // max(price * 100, 1e-12)) * 100
                shares = min(shares, affordable)
                if shares < 100:
                    result = {"code": item.code, "side": item.side, "status": "SKIP_INSUFFICIENT_CASH"}
                    results.append(result)
                    emit({"event": "RESULT", **result})
                    continue
            emit(
                {
                    "event": "SUBMIT_ATTEMPT",
                    "code": item.code,
                    "side": item.side,
                    "shares": shares,
                    "price": float(price),
                }
            )
            try:
                order_id = self.trader.order_stock(
                    self.account,
                    item.code,
                    order_type,
                    shares,
                    xtconstant.FIX_PRICE,
                    float(price),
                    strategy_name,
                    f"{item.side}:{item.reason}",
                )
            except Exception as exc:
                result = {
                    "code": item.code,
                    "side": item.side,
                    "shares": shares,
                    "price": float(price),
                    "order_id": 0,
                    "status": "SUBMIT_EXCEPTION",
                    "error_type": type(exc).__name__,
                }
                results.append(result)
                emit({"event": "RESULT", **result})
                # The remote side-effect is uncertain when order_stock raises. Stop
                # sending any further orders until the operator reconciles broker state.
                break
            result = {
                "code": item.code,
                "side": item.side,
                "shares": shares,
                "price": float(price),
                "order_id": int(order_id),
                "status": "SUBMITTED" if int(order_id) > 0 else "FAILED",
            }
            results.append(result)
            emit({"event": "RESULT", **result})
        return results
