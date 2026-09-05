from __future__ import annotations

from risk.runtime import RuntimeRiskPolicy, evaluate_runtime_risk


def test_runtime_risk_passes_clean_state_without_side_effects():
    report = evaluate_runtime_risk(
        start_of_day_equity=1_000_000,
        current_equity=980_000,
        target_codes=["000001.SZ"],
        position_returns={"000001.SZ": -0.05},
    )
    assert report["passed"] is True
    assert report["side_effects"] == "none"


def test_intraday_drawdown_trips_circuit_breaker():
    report = evaluate_runtime_risk(
        start_of_day_equity=1_000_000,
        current_equity=940_000,
        target_codes=[],
    )
    assert report["passed"] is False
    assert any(item.startswith("intraday_drawdown_circuit_breaker") for item in report["violations"])


def test_blacklist_stop_loss_and_kill_switch_are_fail_closed():
    policy = RuntimeRiskPolicy(
        blacklist_codes=frozenset({"000001.SZ"}),
        max_position_loss=0.10,
        kill_switch=True,
    )
    report = evaluate_runtime_risk(
        start_of_day_equity=1_000_000,
        current_equity=1_000_000,
        target_codes=["000001.SZ"],
        position_returns={"600000.SH": -0.12},
        policy=policy,
    )
    assert report["passed"] is False
    assert "manual_kill_switch" in report["violations"]
    assert "blacklisted_target:000001.SZ" in report["violations"]
    assert any(item.startswith("position_stop_loss:600000.SH") for item in report["violations"])


def test_invalid_equity_state_fails_closed():
    report = evaluate_runtime_risk(
        start_of_day_equity=0,
        current_equity=0,
        target_codes=[],
    )
    assert report["passed"] is False
    assert "invalid_equity_state" in report["violations"]
