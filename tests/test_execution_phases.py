from __future__ import annotations

from qmt_quant.config import CostConfig
from qmt_quant.execution_phases import (
    estimate_buy_cash_reserve,
    split_order_plan,
    validate_sell_position_effect,
)
from qmt_quant.live_trader import OrderInstruction, PositionSnapshot


def test_split_order_plan_preserves_side_ordering() -> None:
    plan = [
        OrderInstruction("000001.SZ", "SELL", 100, 10.0, "reduce"),
        OrderInstruction("000002.SZ", "SELL", 200, 20.0, "reduce"),
        OrderInstruction("000003.SZ", "BUY", 300, 30.0, "increase"),
    ]
    phases = split_order_plan(plan)
    assert [item.code for item in phases.sells] == ["000001.SZ", "000002.SZ"]
    assert [item.code for item in phases.buys] == ["000003.SZ"]


def test_sell_position_effect_requires_full_observed_share_reduction() -> None:
    before = {
        "000001.SZ": PositionSnapshot("000001.SZ", 500, 500),
        "000002.SZ": PositionSnapshot("000002.SZ", 300, 300),
    }
    results = [
        {
            "code": "000001.SZ",
            "side": "SELL",
            "shares": 200,
            "status": "SUBMITTED",
            "order_id": 7,
        }
    ]
    passed = validate_sell_position_effect(
        before,
        results,
        {
            "000001.SZ": PositionSnapshot("000001.SZ", 300, 300),
            "000002.SZ": PositionSnapshot("000002.SZ", 300, 300),
        },
    )
    assert passed["passed"] is True

    failed = validate_sell_position_effect(
        before,
        results,
        {
            "000001.SZ": PositionSnapshot("000001.SZ", 400, 400),
            "000002.SZ": PositionSnapshot("000002.SZ", 300, 300),
        },
    )
    assert failed["passed"] is False
    assert failed["mismatches"][0]["expected_volume"] == 300


def test_sell_position_effect_blocks_unrelated_external_position_drift() -> None:
    before = {
        "000001.SZ": PositionSnapshot("000001.SZ", 500, 500),
        "000002.SZ": PositionSnapshot("000002.SZ", 300, 300),
    }
    results = [
        {
            "code": "000001.SZ",
            "side": "SELL",
            "shares": 100,
            "status": "SUBMITTED",
            "order_id": 7,
        }
    ]
    report = validate_sell_position_effect(
        before,
        results,
        {
            "000001.SZ": PositionSnapshot("000001.SZ", 400, 400),
            "000002.SZ": PositionSnapshot("000002.SZ", 200, 200),
        },
    )
    assert report["passed"] is False
    assert any(item["code"] == "000002.SZ" for item in report["mismatches"])


def test_buy_cash_reserve_includes_minimum_commission() -> None:
    reserve = estimate_buy_cash_reserve(
        [OrderInstruction("000001.SZ", "BUY", 100, 1.0, "increase")],
        cost=CostConfig(commission_rate=0.00025, min_commission=5.0, lot_size=100),
    )
    assert reserve["estimated_buy_cash_required"] == 105.0
    assert reserve["orders"][0]["commission"] == 5.0
