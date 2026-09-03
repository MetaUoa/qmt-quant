from __future__ import annotations

import pandas as pd

from prepare_free_data_shard import select_stock_shard
from qmt_quant.windowed import context_start_for_window, run_window_backtest


def test_shards_are_disjoint_and_cover_all_symbols():
    basic = pd.DataFrame({"ts_code": [f"{i:06d}.SZ" for i in range(23)]})
    parts = [select_stock_shard(basic, i, 5) for i in range(5)]
    sets = [set(part["ts_code"]) for part in parts]
    assert set().union(*sets) == set(basic["ts_code"])
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j])


def test_windowed_backtest_resets_at_requested_period(
    synthetic_bars, permissive_strategy, low_costs
):
    context_start, actual_start, truncated = context_start_for_window(
        synthetic_bars, "000905.SH", permissive_strategy, "2021-01-01"
    )
    assert context_start < actual_start
    assert truncated is False

    result = run_window_backtest(
        synthetic_bars,
        "000905.SH",
        permissive_strategy,
        low_costs,
        trade_start="2021-01-01",
        trade_end="2021-12-31",
    )
    assert result.equity.index.min() >= pd.Timestamp("2021-01-01")
    assert result.equity.index.max() <= pd.Timestamp("2021-12-31")
    if not result.trades.empty:
        assert pd.to_datetime(result.trades["date"]).min() >= pd.Timestamp("2021-01-01")
    assert result.metrics["warmup_truncated"] is False


def test_windowed_backtest_does_not_include_future_dates(
    synthetic_bars, permissive_strategy, low_costs
):
    result = run_window_backtest(
        synthetic_bars,
        "000905.SH",
        permissive_strategy,
        low_costs,
        trade_start="2020-01-01",
        trade_end="2020-06-30",
    )
    assert result.equity.index.max() <= pd.Timestamp("2020-06-30")
    assert result.metrics["end"] <= "2020-06-30"
