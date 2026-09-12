from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

from .transaction_costs import AshareFeeSchedule


SUPPORTED_START = date(2015, 8, 1)
TRANSFER_FEE_CUTOVER = date(2022, 4, 29)
AUG_2023_FEE_CUTOVER = date(2023, 8, 28)


@dataclass(frozen=True)
class HistoricalFeeRegime:
    effective_from: date
    transfer_fee_rate: float
    stamp_tax_rate: float
    exchange_handling_rate: float
    securities_management_rate: float
    authority_key: str


_REGIMES = (
    HistoricalFeeRegime(
        effective_from=SUPPORTED_START,
        transfer_fee_rate=0.00002,
        stamp_tax_rate=0.001,
        exchange_handling_rate=0.0000487,
        securities_management_rate=0.00002,
        authority_key="cn_a_share_2015_08_01",
    ),
    HistoricalFeeRegime(
        effective_from=TRANSFER_FEE_CUTOVER,
        transfer_fee_rate=0.00001,
        stamp_tax_rate=0.001,
        exchange_handling_rate=0.0000487,
        securities_management_rate=0.00002,
        authority_key="cn_a_share_2022_04_29_transfer_cut",
    ),
    HistoricalFeeRegime(
        effective_from=AUG_2023_FEE_CUTOVER,
        transfer_fee_rate=0.00001,
        stamp_tax_rate=0.0005,
        exchange_handling_rate=0.0000341,
        securities_management_rate=0.00002,
        authority_key="cn_a_share_2023_08_28_tax_exchange_cut",
    ),
)


def fee_regime_for(trade_date: date) -> HistoricalFeeRegime:
    if trade_date < SUPPORTED_START:
        raise RuntimeError(
            f"historical A-share fee regime before {SUPPORTED_START.isoformat()} is not encoded"
        )
    selected = _REGIMES[0]
    for regime in _REGIMES:
        if trade_date >= regime.effective_from:
            selected = regime
        else:
            break
    return selected


def historical_fee_schedule(
    trade_date: date,
    *,
    broker_commission_rate: float,
    min_broker_commission: float,
    commission_includes_exchange_and_management: bool = True,
) -> AshareFeeSchedule:
    commission = float(broker_commission_rate)
    minimum = float(min_broker_commission)
    if not math.isfinite(commission) or commission < 0:
        raise ValueError("broker_commission_rate must be finite and non-negative")
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("min_broker_commission must be finite and non-negative")
    regime = fee_regime_for(trade_date)
    return AshareFeeSchedule(
        broker_commission_rate=commission,
        min_broker_commission=minimum,
        transfer_fee_rate=regime.transfer_fee_rate,
        stamp_tax_rate=regime.stamp_tax_rate,
        exchange_handling_rate=regime.exchange_handling_rate,
        securities_management_rate=regime.securities_management_rate,
        commission_includes_exchange_and_management=bool(
            commission_includes_exchange_and_management
        ),
    )


def historical_fee_manifest() -> dict[str, object]:
    return {
        "schema": "qmt-a-share-historical-fees-v1",
        "supported_start": SUPPORTED_START.isoformat(),
        "regimes": [
            {
                "effective_from": regime.effective_from.isoformat(),
                "transfer_fee_rate": regime.transfer_fee_rate,
                "stamp_tax_rate": regime.stamp_tax_rate,
                "exchange_handling_rate": regime.exchange_handling_rate,
                "securities_management_rate": regime.securities_management_rate,
                "authority_key": regime.authority_key,
            }
            for regime in _REGIMES
        ],
        "notes": {
            "broker_commission": "account-specific; caller must supply the research assumption",
            "commission_embedding": "caller explicitly states whether exchange/management fees are embedded",
            "scope": "ordinary Shanghai/Shenzhen A-share auction accounting from 2015-08-01 onward",
        },
    }
