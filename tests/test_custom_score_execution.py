from __future__ import annotations

import pandas as pd

from qmt_quant.backtest import run_backtest
from qmt_quant.windowed import run_window_backtest


def _score_panel(synthetic_bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = synthetic_bars["000905.SH"].index
    return pd.DataFrame(
        {
            "AAA.SZ": 0.0,
            "BBB.SH": 2.0,
            "CCC.SZ": 1.0,
        },
        index=idx,
    )


def test_default_none_overrides_preserve_v4_path(synthetic_bars, permissive_strategy, low_costs):
    baseline = run_backtest(synthetic_bars, "000905.SH", permissive_strategy, low_costs)
    explicit = run_backtest(
        synthetic_bars,
        "000905.SH",
        permissive_strategy,
        low_costs,
        score_override=None,
        risk_on_override=None,
    )
    pd.testing.assert_frame_equal(baseline.equity, explicit.equity)
    pd.testing.assert_frame_equal(baseline.trades, explicit.trades)
    assert baseline.metrics == explicit.metrics


def test_external_score_controls_ranking_while_existing_execution_engine_runs(
    synthetic_bars, permissive_strategy, low_costs
):
    score = _score_panel(synthetic_bars)
    risk_on = pd.Series(True, index=score.index)
    strategy = permissive_strategy.__class__(
        **{**permissive_strategy.__dict__, "top_n": 1}
    )
    result = run_backtest(
        synthetic_bars,
        "000905.SH",
        strategy,
        low_costs,
        score_override=score,
        risk_on_override=risk_on,
    )
    buys = result.trades.loc[result.trades["side"] == "BUY"]
    assert not buys.empty
    assert set(buys["code"]) == {"BBB.SH"}
    assert result.metrics["score_override"] is True
    assert result.metrics["risk_on_override"] is True


def test_missing_or_false_risk_override_is_fail_closed(
    synthetic_bars, permissive_strategy, low_costs
):
    score = _score_panel(synthetic_bars)
    risk_on = pd.Series(False, index=score.index)
    result = run_backtest(
        synthetic_bars,
        "000905.SH",
        permissive_strategy,
        low_costs,
        score_override=score,
        risk_on_override=risk_on,
    )
    assert result.trades.empty


def test_windowed_backtest_slices_external_scores_without_future_use(
    synthetic_bars, permissive_strategy, low_costs
):
    score = _score_panel(synthetic_bars)
    risk_on = pd.Series(True, index=score.index)
    strategy = permissive_strategy.__class__(
        **{**permissive_strategy.__dict__, "top_n": 1}
    )
    result = run_window_backtest(
        synthetic_bars,
        "000905.SH",
        strategy,
        low_costs,
        trade_start="2021-01-01",
        trade_end="2021-12-31",
        score_override=score,
        risk_on_override=risk_on,
    )
    buys = result.trades.loc[result.trades["side"] == "BUY"]
    assert not buys.empty
    assert set(buys["code"]) == {"BBB.SH"}
    assert pd.to_datetime(result.trades["signal_date"]).max() <= pd.Timestamp("2021-12-31")
