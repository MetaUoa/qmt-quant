from __future__ import annotations

import pytest

from qmt_quant.live_trader import PositionSnapshot
from qmt_quant.target_planning import build_target_weight_plan
from risk.pretrade import validate_pretrade


def test_target_weight_planner_preserves_declared_gross_exposure() -> None:
    plan, effective = build_target_weight_plan(
        {"000001.SZ": 0.20, "000002.SZ": 0.20},
        {"000001.SZ": 10.0, "000002.SZ": 20.0},
        {},
        total_asset=100_000.0,
    )
    assert sum(effective.values()) == pytest.approx(0.40)
    buys = {row.code: row.shares for row in plan if row.side == "BUY"}
    assert buys == {"000001.SZ": 2000, "000002.SZ": 1000}


def test_exposure_only_scales_targets_downward_without_renormalizing() -> None:
    _plan, effective = build_target_weight_plan(
        {"000001.SZ": 0.30, "000002.SZ": 0.10},
        {"000001.SZ": 10.0, "000002.SZ": 10.0},
        {},
        total_asset=100_000.0,
        exposure=0.5,
    )
    assert effective == pytest.approx({"000001.SZ": 0.15, "000002.SZ": 0.05})
    assert sum(effective.values()) == pytest.approx(0.20)


def test_missing_target_quote_fails_closed_without_redistribution() -> None:
    with pytest.raises(RuntimeError, match="missing executable prices"):
        build_target_weight_plan(
            {"000001.SZ": 0.20, "000002.SZ": 0.20},
            {"000001.SZ": 10.0},
            {},
            total_asset=100_000.0,
        )


def test_missing_held_position_quote_fails_closed() -> None:
    positions = {
        "600000.SH": PositionSnapshot("600000.SH", volume=1000, available=1000)
    }
    with pytest.raises(RuntimeError, match="600000.SH"):
        build_target_weight_plan(
            {},
            {},
            positions,
            total_asset=100_000.0,
        )


def test_pretrade_concentration_checks_real_target_weights() -> None:
    plan, effective = build_target_weight_plan(
        {"000001.SZ": 0.30, "000002.SZ": 0.10},
        {"000001.SZ": 10.0, "000002.SZ": 10.0},
        {},
        total_asset=100_000.0,
    )
    report = validate_pretrade(
        plan,
        total_asset=100_000.0,
        target_count=2,
        target_weights=effective,
    )
    assert report["passed"] is False
    assert any(item.startswith("target_concentration_too_high:000001.SZ") for item in report["violations"])
