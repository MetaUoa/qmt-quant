from __future__ import annotations

import math
from typing import Mapping

from .live_trader import OrderInstruction, PositionSnapshot


def validate_target_weights(target_weights: Mapping[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for code, raw_weight in target_weights.items():
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
            raise ValueError(f"invalid target weight for {code}: {raw_weight!r}")
        normalized[str(code)] = weight
    total = float(sum(normalized.values()))
    if total > 1.000001:
        raise ValueError(f"target weights exceed 100% gross exposure: {total:.6f}")
    return normalized


def build_target_weight_plan(
    target_weights: Mapping[str, float],
    prices: Mapping[str, float],
    positions: Mapping[str, PositionSnapshot],
    *,
    total_asset: float,
    exposure: float = 1.0,
    lot_size: int = 100,
) -> tuple[list[OrderInstruction], dict[str, float]]:
    """Build sell-first/buy-second orders from explicit portfolio target weights.

    Missing prices fail closed. The planner never redistributes an unpriced target's
    weight to the remaining names. ``exposure`` may only scale weights downward from
    the target bundle; it never renormalizes them upward.
    """
    if not math.isfinite(float(total_asset)) or float(total_asset) <= 0.0:
        raise ValueError("total_asset must be finite and positive")
    scale = float(exposure)
    if not math.isfinite(scale) or scale < 0.0 or scale > 1.0:
        raise ValueError("exposure must be a finite value in [0, 1]")
    lot = int(lot_size)
    if lot <= 0:
        raise ValueError("lot_size must be positive")

    raw_targets = validate_target_weights(target_weights)
    effective = {code: weight * scale for code, weight in raw_targets.items()}

    required_price_codes = {
        code for code, weight in effective.items() if weight > 0.0
    }.union({str(code) for code, pos in positions.items() if int(pos.volume) > 0})
    missing_prices = sorted(
        code
        for code in required_price_codes
        if code not in prices
        or not math.isfinite(float(prices[code]))
        or float(prices[code]) <= 0.0
    )
    if missing_prices:
        raise RuntimeError(
            "missing executable prices for target/held codes: " + ", ".join(missing_prices)
        )

    desired: dict[str, int] = {}
    for code, weight in effective.items():
        if weight <= 0.0:
            desired[code] = 0
            continue
        px = float(prices[code])
        target_value = float(total_asset) * float(weight)
        desired[code] = max(int(target_value // (px * lot)), 0) * lot

    orders: list[OrderInstruction] = []
    for code, pos in sorted(positions.items()):
        current = int(pos.volume)
        target = int(desired.get(code, 0))
        quantity = min(max(current - target, 0), max(int(pos.available), 0))
        quantity = quantity // lot * lot
        if quantity >= lot:
            orders.append(
                OrderInstruction(
                    str(code),
                    "SELL",
                    quantity,
                    float(prices[str(code)]),
                    "rebalance_reduce",
                )
            )

    for code, weight in effective.items():
        if weight <= 0.0:
            continue
        current = int(positions.get(code, PositionSnapshot(code, 0, 0)).volume)
        target = int(desired.get(code, 0))
        quantity = max(target - current, 0)
        quantity = quantity // lot * lot
        if quantity >= lot:
            orders.append(
                OrderInstruction(
                    code,
                    "BUY",
                    quantity,
                    float(prices[code]),
                    "rebalance_increase",
                )
            )

    return orders, effective
