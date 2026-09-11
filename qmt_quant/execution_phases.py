from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

from .backtest_execution import commission
from .config import CostConfig
from .live_trader import OrderInstruction, PositionSnapshot
from .transaction_costs import AshareFeeSchedule, fee_breakdown


@dataclass(frozen=True)
class PhasePlan:
    sells: tuple[OrderInstruction, ...]
    buys: tuple[OrderInstruction, ...]


def _strict_int(value: object, *, name: str, default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be a finite integer")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return int(default)
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    raise ValueError(f"{name} must be an integer")


def split_order_plan(plan: Sequence[OrderInstruction]) -> PhasePlan:
    sells = tuple(item for item in plan if item.side == "SELL")
    buys = tuple(item for item in plan if item.side == "BUY")
    unknown = [item.side for item in plan if item.side not in {"SELL", "BUY"}]
    if unknown:
        raise ValueError(f"unsupported order side(s): {unknown}")
    return PhasePlan(sells=sells, buys=buys)


def submitted_order_ids(results: Sequence[Mapping[str, object]]) -> list[int]:
    ids: set[int] = set()
    for row in results:
        if str(row.get("status", "")) != "SUBMITTED":
            continue
        order_id = _strict_int(row.get("order_id"), name="order_id")
        if order_id > 0:
            ids.add(order_id)
    return sorted(ids)


def validate_full_fill_reconciliation(
    results: Sequence[Mapping[str, object]],
    reconciliation: Mapping[str, object],
) -> dict:
    """Require every submitted instruction to be fully filled by broker history."""
    rows_value = reconciliation.get("orders", [])
    if not isinstance(rows_value, list):
        raise ValueError("reconciliation orders must be a list")
    broker_by_id: dict[int, Mapping[str, object]] = {}
    for raw in rows_value:
        if not isinstance(raw, Mapping):
            raise ValueError("reconciliation order row must be an object")
        order_id = _strict_int(raw.get("order_id"), name="broker.order_id")
        if order_id <= 0:
            continue
        if order_id in broker_by_id:
            raise ValueError(f"duplicate broker order id in reconciliation: {order_id}")
        broker_by_id[order_id] = raw

    mismatches: list[dict[str, object]] = []
    for result in results:
        if str(result.get("status", "")) != "SUBMITTED":
            continue
        order_id = _strict_int(result.get("order_id"), name="result.order_id")
        expected_shares = _strict_int(result.get("shares"), name=f"result.shares:{order_id}")
        expected_code = str(result.get("code", ""))
        broker = broker_by_id.get(order_id)
        if broker is None:
            mismatches.append(
                {"order_id": order_id, "code": expected_code, "reason": "missing_broker_order"}
            )
            continue
        broker_code = str(broker.get("code", ""))
        order_volume = _strict_int(broker.get("order_volume"), name=f"broker.order_volume:{order_id}")
        traded_volume = _strict_int(
            broker.get("traded_volume"), name=f"broker.traded_volume:{order_id}"
        )
        reasons: list[str] = []
        if broker_code != expected_code:
            reasons.append("code_mismatch")
        if order_volume != expected_shares:
            reasons.append("order_volume_mismatch")
        if traded_volume != expected_shares:
            reasons.append("not_fully_filled")
        if reasons:
            mismatches.append(
                {
                    "order_id": order_id,
                    "code": expected_code,
                    "expected_shares": expected_shares,
                    "broker_code": broker_code,
                    "order_volume": order_volume,
                    "traded_volume": traded_volume,
                    "reasons": reasons,
                }
            )
    return {"passed": not mismatches, "mismatches": mismatches}


def has_uncertain_submission(results: Sequence[Mapping[str, object]]) -> bool:
    return any(str(row.get("status", "")) == "SUBMIT_EXCEPTION" for row in results)


def incomplete_results(results: Sequence[Mapping[str, object]]) -> list[dict]:
    return [dict(row) for row in results if str(row.get("status", "")) != "SUBMITTED"]


def expected_positions_after_full_orders(
    before: Mapping[str, PositionSnapshot],
    results: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    expected = {str(code): int(position.volume) for code, position in before.items()}
    for row in results:
        if str(row.get("status", "")) != "SUBMITTED":
            continue
        code = str(row.get("code", ""))
        side = str(row.get("side", ""))
        shares = _strict_int(row.get("shares"), name=f"shares:{code}")
        if side == "SELL":
            expected[code] = max(expected.get(code, 0) - shares, 0)
        elif side == "BUY":
            expected[code] = expected.get(code, 0) + shares
        else:
            raise ValueError(f"unsupported submitted side during position validation: {side}")
    return expected


def validate_position_effect(
    before: Mapping[str, PositionSnapshot],
    results: Sequence[Mapping[str, object]],
    after: Mapping[str, PositionSnapshot],
) -> dict:
    """Require the whole account share map to match the submitted full-fill effects."""
    expected = expected_positions_after_full_orders(before, results)
    observed = {str(code): int(position.volume) for code, position in after.items()}
    mismatches: list[dict[str, object]] = []
    for code in sorted(set(expected) | set(observed)):
        expected_volume = int(expected.get(code, 0))
        observed_volume = int(observed.get(code, 0))
        if observed_volume != expected_volume:
            mismatches.append(
                {
                    "code": code,
                    "expected_volume": expected_volume,
                    "observed_volume": observed_volume,
                }
            )
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "expected_positions": expected,
        "observed_positions": observed,
    }


def validate_sell_position_effect(
    before: Mapping[str, PositionSnapshot],
    sell_results: Sequence[Mapping[str, object]],
    after: Mapping[str, PositionSnapshot],
) -> dict:
    return validate_position_effect(before, sell_results, after)


def estimate_buy_cash_reserve(
    plan: Sequence[OrderInstruction],
    *,
    cost: CostConfig,
    fee_schedule: AshareFeeSchedule | None = None,
) -> dict:
    rows: list[dict[str, object]] = []
    total = 0.0
    for item in plan:
        if item.side != "BUY":
            continue
        notional = float(item.shares) * float(item.reference_price)
        if fee_schedule is None:
            fee = commission(cost, notional)
            reserve = notional + fee
            fee_payload: dict[str, object] = {"commission": float(fee)}
        else:
            breakdown = fee_breakdown(side="BUY", notional=notional, schedule=fee_schedule)
            fee = float(breakdown.broker_commission)
            reserve = notional + float(breakdown.total_cash_fee)
            fee_payload = asdict(breakdown)
        total += reserve
        rows.append(
            {
                "code": item.code,
                "shares": int(item.shares),
                "reference_price": float(item.reference_price),
                "notional": float(notional),
                "commission": float(fee),
                "fees": fee_payload,
                "reserve": float(reserve),
            }
        )
    payload = {
        "estimated_buy_cash_required": float(total),
        "orders": rows,
        "cost": asdict(cost),
    }
    if fee_schedule is not None:
        payload["ashare_fee_schedule"] = asdict(fee_schedule)
    return payload
