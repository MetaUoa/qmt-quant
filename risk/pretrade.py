from __future__ import annotations

from dataclasses import asdict, dataclass

from qmt_quant.live_trader import OrderInstruction


@dataclass(frozen=True)
class PretradeLimits:
    max_orders: int = 40
    max_single_order_asset_fraction: float = 0.30
    max_single_target_weight: float = 0.25
    min_total_asset: float = 10_000.0


def validate_pretrade(
    plan: list[OrderInstruction],
    *,
    total_asset: float,
    target_count: int,
    limits: PretradeLimits | None = None,
) -> dict:
    limits = limits or PretradeLimits()
    violations: list[str] = []
    if total_asset < limits.min_total_asset:
        violations.append("total_asset_below_minimum")
    if len(plan) > limits.max_orders:
        violations.append("too_many_orders")
    if target_count > 0 and (1.0 / target_count) > limits.max_single_target_weight:
        violations.append("target_concentration_too_high")
    for item in plan:
        notional = float(item.shares) * float(item.reference_price)
        fraction = notional / total_asset if total_asset > 0 else 1.0
        if fraction > limits.max_single_order_asset_fraction:
            violations.append(f"single_order_too_large:{item.code}:{fraction:.4f}")
    return {
        "passed": not violations,
        "violations": violations,
        "order_count": len(plan),
        "target_count": int(target_count),
        "total_asset": float(total_asset),
        "limits": asdict(limits),
    }
