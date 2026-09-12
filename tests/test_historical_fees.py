from __future__ import annotations

from datetime import date

import pytest

from qmt_quant.historical_fees import fee_regime_for, historical_fee_schedule


def test_transfer_fee_cutover_2022_04_29() -> None:
    before = fee_regime_for(date(2022, 4, 28))
    after = fee_regime_for(date(2022, 4, 29))
    assert before.transfer_fee_rate == pytest.approx(0.00002)
    assert after.transfer_fee_rate == pytest.approx(0.00001)
    assert before.stamp_tax_rate == pytest.approx(0.001)
    assert after.stamp_tax_rate == pytest.approx(0.001)


def test_stamp_tax_and_exchange_fee_cutover_2023_08_28() -> None:
    before = fee_regime_for(date(2023, 8, 27))
    after = fee_regime_for(date(2023, 8, 28))
    assert before.stamp_tax_rate == pytest.approx(0.001)
    assert after.stamp_tax_rate == pytest.approx(0.0005)
    assert before.exchange_handling_rate == pytest.approx(0.0000487)
    assert after.exchange_handling_rate == pytest.approx(0.0000341)
    assert after.securities_management_rate == pytest.approx(0.00002)


def test_schedule_keeps_broker_commission_explicit() -> None:
    schedule = historical_fee_schedule(
        date(2024, 1, 2),
        broker_commission_rate=0.00018,
        min_broker_commission=5.0,
        commission_includes_exchange_and_management=True,
    )
    assert schedule.broker_commission_rate == pytest.approx(0.00018)
    assert schedule.min_broker_commission == pytest.approx(5.0)
    assert schedule.transfer_fee_rate == pytest.approx(0.00001)
    assert schedule.stamp_tax_rate == pytest.approx(0.0005)


def test_pre_supported_fee_history_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not encoded"):
        fee_regime_for(date(2015, 7, 31))
