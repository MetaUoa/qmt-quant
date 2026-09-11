from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

from .backtest_execution import commission
from .config import CostConfig
from .live_trader import OrderInstruction, PositionSnapshot


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


def has_uncertain_submission(results: Sequence[Mapping[str, object]]) -> bool:
    return any(str(row.get("status", "")) == "SUBMIT_EXCEPTION" for row in results)


def incomplete_results(results: Sequence[Mapping[str, object]]) -> list[dict]:
    return [dict(row) for row in results if str(row.get("status", "")) != "SUBMITTED"]


def expected_positions_after_full_sells(
    before: Mapping[str, PositionSnapshot],
    sell_results: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    expected = {str(code): int(position.volume) for code, position in before.items()}
    for row in sell_results:
        if str(row.get("status", "")) != "SUBMITTED":
            continue
        code = str(row.get("code", ""))
        shares = _strict_int(row.get("shares"), name=f"shares:{code}")
        expected[code] = max(expected.get(code, 0) - shares, 0)
    return expected


def validate_sell_position_effect(
    before: Mapping[str, PositionSnapshot],
    sell_results: Sequence[Mapping[str, object]],
    after: Mapping[str, PositionSnapshot],
) -> dict:
    """Require the post-sell snapshot to equal the expected full-fill share state.

    This checks every observed position, not just sold symbols. An unrelated manual or
    external trade during the live batch therefore blocks the BUY phase rather than
    silently changing the account beneath the executor.
    """
    expected = expected_positions_after_full_sells(before, sell_results)
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


def estimate_buy_cash_reserve(
    plan: Sequence[OrderInstruction],
    *,
    cost: CostConfig,
) -> dict:
    rows: list[dict[str, object]] = []
    total = 0.0
    for item in plan:
        if item.side != "BUY":
            continue
        notional = float(item.shares) * float(item.reference_price)
        fee = commission(cost, notional)
        reserve = notional + fee
        total += reserve
        rows.append(
            {
                "code": item.code,
                "shares": int(item.shares),
                "reference_price": float(item.reference_price),
                "notional": float(notional),
                "commission": float(fee),
                "reserve": float(reserve),
            }
        )
    return {
        "estimated_buy_cash_required": float(total),
        "orders": rows,
        "cost": asdict(cost),
    }
