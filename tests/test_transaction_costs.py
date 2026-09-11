from __future__ import annotations

import pytest

from qmt_quant.transaction_costs import (
    AshareFeeSchedule,
    affordable_buy_quantity_with_fees,
    fee_breakdown,
    reconcile_phase_cash,
)


def test_default_fee_schedule_avoids_double_counting_embedded_regulatory_fees() -> None:
    schedule = AshareFeeSchedule()
    buy = fee_breakdown(side="BUY", notional=10_000.0, schedule=schedule)
    sell = fee_breakdown(side="SELL", notional=10_000.0, schedule=schedule)

    assert buy.broker_commission == pytest.approx(5.0)
    assert buy.transfer_fee == pytest.approx(0.1)
    assert buy.stamp_tax == 0.0
    assert buy.exchange_handling_fee == pytest.approx(0.341)
    assert buy.securities_management_fee == pytest.approx(0.2)
    assert buy.embedded_regulatory_fee == pytest.approx(0.541)
    assert buy.separately_charged_regulatory_fee == 0.0
    assert buy.total_cash_fee == pytest.approx(5.1)

    assert sell.stamp_tax == pytest.approx(5.0)
    assert sell.total_cash_fee == pytest.approx(10.1)


def test_regulatory_fees_can_be_explicitly_added_when_broker_commission_excludes_them() -> None:
    schedule = AshareFeeSchedule(commission_includes_exchange_and_management=False)
    buy = fee_breakdown(side="BUY", notional=10_000.0, schedule=schedule)
    assert buy.embedded_regulatory_fee == 0.0
    assert buy.separately_charged_regulatory_fee == pytest.approx(0.541)
    assert buy.total_cash_fee == pytest.approx(5.641)


def test_buy_affordability_includes_transfer_fee_and_minimum_commission() -> None:
    schedule = AshareFeeSchedule()
    assert (
        affordable_buy_quantity_with_fees(
            requested_shares=100,
            execution_price=10.0,
            cash=1005.0,
            lot_size=100,
            schedule=schedule,
        )
        == 0
    )
    assert (
        affordable_buy_quantity_with_fees(
            requested_shares=100,
            execution_price=10.0,
            cash=1005.01,
            lot_size=100,
            schedule=schedule,
        )
        == 100
    )


def test_sell_cash_reconciliation_matches_exact_trade_and_modeled_fees() -> None:
    schedule = AshareFeeSchedule()
    results = [
        {
            "status": "SUBMITTED",
            "side": "SELL",
            "order_id": 7,
            "code": "000001.SZ",
            "shares": 1000,
        }
    ]
    trades = [
        {
            "order_id": 7,
            "traded_volume": 1000,
            "traded_price": 10.0,
        }
    ]
    # 10,000 gross - (5 commission + 0.1 transfer + 5 stamp) = 9,989.9 net cash.
    report = reconcile_phase_cash(
        side="SELL",
        cash_before=1000.0,
        cash_after=10_989.9,
        results=results,
        trades=trades,
        schedule=schedule,
        absolute_tolerance=0.01,
        relative_tolerance_bps=0.0,
    )
    assert report["passed"] is True
    assert report["gross_traded_notional"] == pytest.approx(10_000.0)
    assert report["expected_total_cash_fees"] == pytest.approx(10.1)
    assert report["unexplained_cash_delta"] == pytest.approx(0.0)


def test_cash_reconciliation_fails_external_cash_drift_and_missing_fill_volume() -> None:
    schedule = AshareFeeSchedule()
    results = [
        {
            "status": "SUBMITTED",
            "side": "BUY",
            "order_id": 8,
            "code": "000002.SZ",
            "shares": 100,
        }
    ]
    trades = [{"order_id": 8, "traded_volume": 50, "traded_price": 10.0}]
    report = reconcile_phase_cash(
        side="BUY",
        cash_before=10_000.0,
        cash_after=9_400.0,
        results=results,
        trades=trades,
        schedule=schedule,
        absolute_tolerance=1.0,
        relative_tolerance_bps=0.0,
    )
    assert report["passed"] is False
    reasons = str(report["violations"])
    assert "trade_volume_mismatch" in reasons
    assert "cash_effect_mismatch" in reasons
