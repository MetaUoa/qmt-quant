from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .config import CostConfig
from .backtest_execution import commission
from .live_trader import OrderInstruction, PositionSnapshot


@dataclass(frozen=True)
class PhasePlan:
    sells: tuple[OrderInstruction, ...]
    buys: tuple[OrderInstruction, ...]


def split_order_plan(plan: Sequence[OrderInstruction]) -> PhasePlan:
    sells = tuple(item for item in plan if item.side == "SELL")
    buys = tuple(item for item in plan if item.side == "BUY")
    unknown = [item.side for item in plan if item.side not in {"SELL", "BUY"}]
    if unknown:
        raise ValueError(f"unsupported order side(s): {unknown}")
    return PhasePlan(sells=sells, buys=buys)


def submitted_order_ids(results: Sequence[Mapping[str, object]]) -> list[int]:
    return sorted(
        {
            int(row.get("order_id", 0) or 0)
            for row in results
            if str(row.get("status", "")) == "SUBMITTED" and int(row.get("order_id", 0) or 0) > 0
        }
    )


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
        shares = int(row.get("shares", 0) or 0)
        expected[code] = max(expected.get(code, 0) - shares, 0)
    return expected


def validate_sell_position_effect(
    before: Mapping[str, PositionSnapshot],
    sell_results: Sequence[Mapping[str, object]],
    after: Mapping[str, PositionSnapshot],
) -> dict:
    expected = expected_positions_after_full_sells(before, sell_results)
    mismatches: list[dict[str, object]] = []
    touched_codes = {
        str(row.get("code", ""))
        for row in sell_results
        if str(row.get("status", "")) == "SUBMITTED"
    }
    for code in sorted(touched_codes):
        expected_volume = int(expected.get(code, 0))
        observed_volume = int(after.get(code, PositionSnapshot(code, 0, 0)).volume)
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
        "expected_positions": {code: expected[code] for code in sorted(touched_codes)},
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
