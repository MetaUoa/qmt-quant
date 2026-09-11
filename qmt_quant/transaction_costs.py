from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

from .config import CostConfig


@dataclass(frozen=True)
class AshareFeeSchedule:
    """Investor cash-fee model for ordinary A-share auction trades.

    By default, broker commission is treated as an all-in commission that already
    embeds exchange handling and securities-management fees. Transfer fee and
    sell-side stamp tax remain separate investor cash charges.
    """

    broker_commission_rate: float = 0.00025
    min_broker_commission: float = 5.0
    transfer_fee_rate: float = 0.00001
    stamp_tax_rate: float = 0.0005
    exchange_handling_rate: float = 0.0000341
    securities_management_rate: float = 0.00002
    commission_includes_exchange_and_management: bool = True

    @classmethod
    def from_cost_config(
        cls,
        cost: CostConfig,
        *,
        transfer_fee_rate: float = 0.00001,
        stamp_tax_rate: float = 0.0005,
        exchange_handling_rate: float = 0.0000341,
        securities_management_rate: float = 0.00002,
        commission_includes_exchange_and_management: bool = True,
    ) -> "AshareFeeSchedule":
        return cls(
            broker_commission_rate=float(cost.commission_rate),
            min_broker_commission=float(cost.min_commission),
            transfer_fee_rate=float(transfer_fee_rate),
            stamp_tax_rate=float(stamp_tax_rate),
            exchange_handling_rate=float(exchange_handling_rate),
            securities_management_rate=float(securities_management_rate),
            commission_includes_exchange_and_management=bool(
                commission_includes_exchange_and_management
            ),
        )

    def validate(self) -> None:
        for name in (
            "broker_commission_rate",
            "min_broker_commission",
            "transfer_fee_rate",
            "stamp_tax_rate",
            "exchange_handling_rate",
            "securities_management_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class AshareFeeBreakdown:
    side: str
    notional: float
    broker_commission: float
    transfer_fee: float
    stamp_tax: float
    exchange_handling_fee: float
    securities_management_fee: float
    embedded_regulatory_fee: float
    separately_charged_regulatory_fee: float
    total_cash_fee: float


def fee_breakdown(
    *,
    side: str,
    notional: float,
    schedule: AshareFeeSchedule,
) -> AshareFeeBreakdown:
    schedule.validate()
    side_token = str(side).upper()
    if side_token not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported A-share side: {side}")
    value = float(notional)
    if not math.isfinite(value) or value < 0:
        raise ValueError("notional must be finite and non-negative")
    if value == 0:
        return AshareFeeBreakdown(side_token, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    commission = max(
        float(schedule.min_broker_commission),
        value * float(schedule.broker_commission_rate),
    )
    transfer = value * float(schedule.transfer_fee_rate)
    stamp = value * float(schedule.stamp_tax_rate) if side_token == "SELL" else 0.0
    handling = value * float(schedule.exchange_handling_rate)
    management = value * float(schedule.securities_management_rate)
    regulatory = handling + management
    separately_charged = 0.0 if schedule.commission_includes_exchange_and_management else regulatory
    total = commission + transfer + stamp + separately_charged
    return AshareFeeBreakdown(
        side=side_token,
        notional=value,
        broker_commission=float(commission),
        transfer_fee=float(transfer),
        stamp_tax=float(stamp),
        exchange_handling_fee=float(handling),
        securities_management_fee=float(management),
        embedded_regulatory_fee=float(regulatory if schedule.commission_includes_exchange_and_management else 0.0),
        separately_charged_regulatory_fee=float(separately_charged),
        total_cash_fee=float(total),
    )


def total_cash_required_for_buy(
    *,
    shares: int,
    execution_price: float,
    schedule: AshareFeeSchedule,
) -> float:
    notional = max(int(shares), 0) * float(execution_price)
    fees = fee_breakdown(side="BUY", notional=notional, schedule=schedule)
    return float(notional + fees.total_cash_fee)


def affordable_buy_quantity_with_fees(
    *,
    requested_shares: int,
    execution_price: float,
    cash: float,
    lot_size: int,
    schedule: AshareFeeSchedule,
) -> int:
    qty = max(int(requested_shares), 0)
    lot = int(lot_size)
    if lot <= 0:
        raise ValueError("lot_size must be positive")
    while qty >= lot:
        if total_cash_required_for_buy(
            shares=qty,
            execution_price=float(execution_price),
            schedule=schedule,
        ) <= float(cash):
            return qty
        qty -= lot
    return 0


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    raise ValueError(f"{name} must be an integer")


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must be numeric")
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric") from exc
    else:
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def reconcile_phase_cash(
    *,
    side: str,
    cash_before: float,
    cash_after: float,
    results: Sequence[Mapping[str, object]],
    trades: Sequence[Mapping[str, object]],
    schedule: AshareFeeSchedule,
    absolute_tolerance: float = 2.0,
    relative_tolerance_bps: float = 0.5,
) -> dict[str, object]:
    """Compare broker cash movement with fills and the configured investor fee model.

    Any external deposit/withdrawal, fee mismatch or missing trade history appears as
    unexplained cash and fails closed once outside the configured tolerance.
    """

    side_token = str(side).upper()
    if side_token not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported side: {side}")
    before = float(cash_before)
    after = float(cash_after)
    if not all(math.isfinite(value) and value >= 0 for value in (before, after)):
        raise ValueError("cash snapshots must be finite and non-negative")
    if absolute_tolerance < 0 or relative_tolerance_bps < 0:
        raise ValueError("cash reconciliation tolerances must be non-negative")

    submitted: dict[int, Mapping[str, object]] = {}
    for row in results:
        if str(row.get("status", "")) != "SUBMITTED" or str(row.get("side", "")).upper() != side_token:
            continue
        order_id = _strict_int(row.get("order_id"), name="result.order_id")
        if order_id <= 0 or order_id in submitted:
            raise ValueError("submitted order ids must be unique positive integers")
        submitted[order_id] = row

    trade_notional_by_order = {order_id: 0.0 for order_id in submitted}
    trade_volume_by_order = {order_id: 0 for order_id in submitted}
    for row in trades:
        order_id = _strict_int(row.get("order_id"), name="trade.order_id")
        if order_id not in submitted:
            continue
        volume = _strict_int(row.get("traded_volume"), name=f"trade.traded_volume:{order_id}")
        price = _strict_float(row.get("traded_price", 0.0), name=f"trade.traded_price:{order_id}")
        if volume <= 0 or price <= 0:
            raise ValueError(f"trade {order_id} has invalid volume or price")
        trade_volume_by_order[order_id] += volume
        trade_notional_by_order[order_id] += volume * price

    order_rows: list[dict[str, object]] = []
    gross_notional = 0.0
    expected_fee_total = 0.0
    violations: list[dict[str, object]] = []
    for order_id, result in submitted.items():
        expected_volume = _strict_int(result.get("shares"), name=f"result.shares:{order_id}")
        observed_volume = int(trade_volume_by_order.get(order_id, 0))
        notional = float(trade_notional_by_order.get(order_id, 0.0))
        if observed_volume != expected_volume:
            violations.append(
                {
                    "order_id": order_id,
                    "reason": "trade_volume_mismatch",
                    "expected_volume": expected_volume,
                    "observed_volume": observed_volume,
                }
            )
        fees = fee_breakdown(side=side_token, notional=notional, schedule=schedule)
        gross_notional += notional
        expected_fee_total += fees.total_cash_fee
        order_rows.append(
            {
                "order_id": order_id,
                "code": str(result.get("code", "")),
                "shares": expected_volume,
                "traded_volume": observed_volume,
                "traded_notional": notional,
                "fees": asdict(fees),
            }
        )

    observed_cash_delta = after - before
    expected_cash_delta = (
        gross_notional - expected_fee_total
        if side_token == "SELL"
        else -gross_notional - expected_fee_total
    )
    implied_cash_fee = (
        gross_notional - observed_cash_delta
        if side_token == "SELL"
        else -observed_cash_delta - gross_notional
    )
    unexplained = observed_cash_delta - expected_cash_delta
    tolerance = max(
        float(absolute_tolerance),
        gross_notional * float(relative_tolerance_bps) / 10_000.0,
    )
    if abs(unexplained) > tolerance:
        violations.append(
            {
                "reason": "cash_effect_mismatch",
                "observed_cash_delta": observed_cash_delta,
                "expected_cash_delta": expected_cash_delta,
                "unexplained_cash_delta": unexplained,
                "tolerance": tolerance,
            }
        )

    return {
        "passed": not violations,
        "side": side_token,
        "cash_before": before,
        "cash_after": after,
        "observed_cash_delta": float(observed_cash_delta),
        "gross_traded_notional": float(gross_notional),
        "expected_total_cash_fees": float(expected_fee_total),
        "implied_broker_cash_fees": float(implied_cash_fee),
        "expected_cash_delta": float(expected_cash_delta),
        "unexplained_cash_delta": float(unexplained),
        "tolerance": float(tolerance),
        "fee_schedule": asdict(schedule),
        "orders": order_rows,
        "violations": violations,
    }
