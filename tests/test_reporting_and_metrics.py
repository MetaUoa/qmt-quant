from __future__ import annotations

import json

import pandas as pd
import pytest

from qmt_quant.backtest import BacktestResult, calculate_metrics
from qmt_quant.reporting import save_result, yearly_returns


def test_calculate_metrics_known_equity_curve():
    idx = pd.to_datetime(["2020-01-01", "2020-12-31", "2021-12-31"])
    equity = pd.Series([1.0, 1.5, 2.0], index=idx)
    metrics = calculate_metrics(equity)
    assert metrics["multiple"] == pytest.approx(2.0)
    assert metrics["total_return"] == pytest.approx(1.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_yearly_returns_are_year_local():
    idx = pd.to_datetime(["2020-01-02", "2020-12-31", "2021-01-04", "2021-12-31"])
    equity = pd.Series([100.0, 120.0, 120.0, 90.0], index=idx)
    out = yearly_returns(equity).set_index("year")
    assert out.loc[2020, "return"] == pytest.approx(0.20)
    assert out.loc[2021, "return"] == pytest.approx(-0.25)


def test_save_result_writes_complete_artifact_set(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=5)
    equity = pd.DataFrame(
        {"equity": [1_000_000, 1_010_000, 1_020_000, 1_015_000, 1_030_000], "cash": 0.0},
        index=idx,
    )
    result = BacktestResult(
        equity=equity,
        trades=pd.DataFrame([{"date": idx[1], "code": "AAA.SZ", "side": "BUY"}]),
        metrics={"multiple": 1.03, "strict_reference": True},
        config={"strategy": {"top_n": 8}},
    )
    coverage = pd.DataFrame([{"code": "AAA.SZ", "loaded": True}])
    quality = {"symbol_coverage_ratio": 1.0}
    out = save_result(result, tmp_path, coverage=coverage, data_quality=quality)

    expected = {
        "equity.csv",
        "trades.csv",
        "yearly_returns.csv",
        "metrics.json",
        "config.json",
        "universe_coverage.csv",
        "data_quality.json",
    }
    assert expected.issubset({p.name for p in out.iterdir()})
    with (out / "metrics.json").open(encoding="utf-8") as handle:
        assert json.load(handle)["strict_reference"] is True
