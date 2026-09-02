from __future__ import annotations

import pandas as pd

from qmt_quant.acceptance import grade_strategy
from qmt_quant.backtest import run_backtest
from qmt_quant.config import AcceptanceConfig
from qmt_quant.live_trader import PositionSnapshot, build_equal_weight_plan
from qmt_quant.research import add_neighborhood_stability, config_key, research_score
from qmt_quant.stress import monte_carlo_daily_returns, run_stress_suite, stress_summary


def test_research_score_and_neighborhood_stability(synthetic_bars, permissive_strategy, low_costs):
    result = run_backtest(synthetic_bars, "000905.SH", permissive_strategy, low_costs)
    score, diag = research_score(result, "2019-01-01", "2021-12-31", max_drawdown=0.99)
    assert score != float("-inf")
    assert "trade_concentration" in diag
    key = config_key(permissive_strategy)
    rows = pd.DataFrame([{"candidate": key, "raw_score": score}])
    stable = add_neighborhood_stability(rows, {key: permissive_strategy})
    assert stable.loc[0, "stable_score"] == score


def test_stress_suite_changes_execution_path(synthetic_bars, permissive_strategy, low_costs):
    frame, results = run_stress_suite(
        synthetic_bars,
        "000905.SH",
        permissive_strategy,
        low_costs,
    )
    assert len(frame) >= 10
    assert "execution_delay_2" in results
    assert results["execution_delay_2"].metrics["execution_delay_sessions"] == 2
    assert results["fill_90pct"].metrics["fill_probability"] == 0.90
    summary = stress_summary(frame)
    assert 0.0 <= summary["pass_ratio"] <= 1.0
    mc = monte_carlo_daily_returns(results["base"].equity["equity"], simulations=50, seed=1)
    assert mc["simulations"] == 50
    assert 0.0 <= mc["probability_profitable"] <= 1.0


def test_acceptance_grades_a_when_all_gates_pass():
    metrics = {"multiple": 160.0, "max_drawdown": -0.30, "sharpe": 1.7}
    oos = {"cagr": 0.40}
    folds = pd.DataFrame({"validation_return": [0.1, 0.2, 0.3, -0.05, 0.15]})
    stress = {"pass_ratio": 0.90}
    report = grade_strategy(metrics, oos, folds, stress, AcceptanceConfig())
    assert report["grade"] == "A"


def test_live_order_plan_sells_first_and_uses_board_lots():
    positions = {
        "OLD.SH": PositionSnapshot("OLD.SH", 500, 400, 5000.0),
        "KEEP.SZ": PositionSnapshot("KEEP.SZ", 200, 200, 2000.0),
    }
    prices = {"OLD.SH": 10.0, "KEEP.SZ": 10.0, "NEW.SZ": 20.0}
    plan = build_equal_weight_plan(
        ["KEEP.SZ", "NEW.SZ"],
        prices,
        positions,
        total_asset=20_000.0,
        exposure=1.0,
    )
    assert plan
    sides = [x.side for x in plan]
    assert "SELL" in sides and "BUY" in sides
    assert sides.index("SELL") < sides.index("BUY")
    assert all(x.shares % 100 == 0 for x in plan)


def test_pretrade_risk_gate_rejects_concentrated_single_target():
    from risk.pretrade import validate_pretrade

    positions = {}
    prices = {"AAA.SZ": 10.0}
    plan = build_equal_weight_plan(["AAA.SZ"], prices, positions, total_asset=100_000.0)
    report = validate_pretrade(plan, total_asset=100_000.0, target_count=1)
    assert report["passed"] is False
    assert "target_concentration_too_high" in report["violations"]


def test_latest_signal_uses_current_pit_st_snapshot(synthetic_bars, permissive_strategy):
    from qmt_quant.reference_data import ReferenceData
    from qmt_quant.signals import latest_target_codes

    idx = synthetic_bars["000905.SH"].index
    codes = [c for c in synthetic_bars if c != "000905.SH"]
    basic = pd.DataFrame(
        [
            {
                "ts_code": code,
                "exchange": "SSE" if code.endswith(".SH") else "SZSE",
                "list_date": "20100101",
                "delist_date": None,
            }
            for code in codes
        ]
    )
    st = pd.DataFrame([{"trade_date": idx[-1].strftime("%Y%m%d"), "ts_code": "AAA.SZ"}])
    ref = ReferenceData(basic, idx, st=st, limits=pd.DataFrame())
    ts, selected, diagnostics = latest_target_codes(
        synthetic_bars,
        "000905.SH",
        permissive_strategy,
        reference=ref,
        signal_date=idx[-1],
        raw_bars=synthetic_bars,
        strict_st=True,
    )
    assert ts == idx[-1]
    assert diagnostics["risk_on"] is True
    assert "AAA.SZ" not in selected


def test_two_session_execution_delay_keeps_signal_strictly_prior(synthetic_bars, permissive_strategy, low_costs):
    from dataclasses import replace

    cfg = replace(permissive_strategy, execution_delay_sessions=2)
    result = run_backtest(synthetic_bars, "000905.SH", cfg, low_costs)
    assert result.metrics["execution_delay_sessions"] == 2
    if not result.trades.empty:
        calendar = synthetic_bars["000905.SH"].index
        loc = {ts: i for i, ts in enumerate(calendar)}
        for row in result.trades.itertuples(index=False):
            assert loc[pd.Timestamp(row.date)] - loc[pd.Timestamp(row.signal_date)] == 2


def test_live_tick_guard_marks_missing_order_book_side_untradable():
    from qmt_quant.live_trader import QmtBroker

    ticks = {
        "UP.SZ": {"lastPrice": 10.0, "askPrice": [0.0], "bidPrice": [10.0], "lastClose": 9.1},
        "DOWN.SH": {"lastPrice": 8.0, "askPrice": [8.0], "bidPrice": [0.0], "lastClose": 8.8},
    }
    out = QmtBroker.executable_prices(ticks)
    assert out["UP.SZ"]["buy_tradable"] is False
    assert out["UP.SZ"]["sell_tradable"] is True
    assert out["DOWN.SH"]["buy_tradable"] is True
    assert out["DOWN.SH"]["sell_tradable"] is False
