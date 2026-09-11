from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

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
    target_weights: Mapping[str, float] | None = None,
    limits: PretradeLimits | None = None,
) -> dict:
    limits = limits or PretradeLimits()
    violations: list[str] = []
    if not math.isfinite(float(total_asset)) or total_asset < limits.min_total_asset:
        violations.append("total_asset_below_minimum")
    if len(plan) > limits.max_orders:
        violations.append("too_many_orders")

    normalized_weights: dict[str, float] = {}
    if target_weights is not None:
        for code, raw_weight in target_weights.items():
            weight = float(raw_weight)
            normalized_weights[str(code)] = weight
            if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
                violations.append(f"invalid_target_weight:{code}")
            elif weight > limits.max_single_target_weight:
                violations.append(f"target_concentration_too_high:{code}:{weight:.4f}")
        finite_weights = [w for w in normalized_weights.values() if math.isfinite(w)]
        if sum(finite_weights) > 1.000001:
            violations.append("target_gross_exposure_too_high")
    elif target_count > 0 and (1.0 / target_count) > limits.max_single_target_weight:
        # Legacy equal-weight callers retain the historical concentration check.
        violations.append("target_concentration_too_high")

    for item in plan:
        notional = float(item.shares) * float(item.reference_price)
        fraction = notional / total_asset if total_asset > 0 else 1.0
        if not math.isfinite(fraction) or fraction > limits.max_single_order_asset_fraction:
            violations.append(f"single_order_too_large:{item.code}:{fraction:.4f}")
    return {
        "passed": not violations,
        "violations": violations,
        "order_count": len(plan),
        "target_count": int(target_count),
        "target_weight_sum": (
            float(sum(normalized_weights.values())) if normalized_weights else None
        ),
        "target_weights": normalized_weights,
        "total_asset": float(total_asset),
        "limits": asdict(limits),
    }
