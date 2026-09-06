from __future__ import annotations

from qmt_quant.backtest_reporting import BacktestDiagnostics, assemble_backtest_metrics


def _diagnostics(**overrides) -> BacktestDiagnostics:
    values = {
        "trade_count": 12,
        "rebalance_count": 3,
        "initial_cash": 1_000_000.0,
        "blocked_st_candidates": 1,
        "blocked_limit_buys": 2,
        "blocked_limit_sells": 3,
        "blocked_suspended": 4,
        "blocked_t1_sells": 5,
        "missing_suspend_rows": 0,
        "missing_limit_rows": 0,
        "missing_st_dates": 0,
        "missing_limit_dates": 0,
        "point_in_time_universe": True,
        "strict_reference": True,
        "raw_limit_reference": True,
        "blocked_random_fill": 6,
        "execution_delay_sessions": 1,
        "fill_probability": 1.0,
        "average_market_breadth": 0.55,
    }
    values.update(overrides)
    return BacktestDiagnostics(**values)


def test_metrics_assembly_preserves_existing_public_contract() -> None:
    metrics = assemble_backtest_metrics(
        {"total_return": 0.25, "sharpe": 1.1},
        _diagnostics(),
    )
    assert metrics["total_return"] == 0.25
    assert metrics["sharpe"] == 1.1
    assert metrics["trade_count"] == 12
    assert metrics["rebalance_count"] == 3
    assert metrics["blocked_t1_sells"] == 5
    assert metrics["missing_suspend_rows"] == 0
    assert metrics["strict_reference"] is True
    assert metrics["t_plus_one_enforced"] is True
    assert metrics["limit_model"] == "open_auction_reference_plus_one_price_daily_fallback"
    assert metrics["intraday_limit_touch_modelled"] is False
    assert metrics["average_market_breadth"] == 0.55
    assert "score_override" not in metrics
    assert "risk_on_override" not in metrics


def test_override_markers_are_only_emitted_when_active() -> None:
    metrics = assemble_backtest_metrics(
        {},
        _diagnostics(score_override=True, risk_on_override=True),
    )
    assert metrics["score_override"] is True
    assert metrics["risk_on_override"] is True
