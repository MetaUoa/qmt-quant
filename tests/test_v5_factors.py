import numpy as np
import pandas as pd

from qmt_quant.factors import (
    V5FactorConfig,
    build_v5_raw_factors,
    combine_ranked_factors,
    cross_sectional_rank,
)


def _sample_panels(periods: int = 180) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    idx = pd.bdate_range("2020-01-01", periods=periods)
    base = np.arange(periods, dtype=float)
    close = pd.DataFrame(
        {
            "AAA.SZ": 10.0 + base * 0.08,
            "BBB.SH": 12.0 + base * 0.03,
            "CCC.SZ": 15.0 - base * 0.01,
        },
        index=idx,
    )
    amount = pd.DataFrame(
        {
            "AAA.SZ": 50_000_000.0 + base * 20_000.0,
            "BBB.SH": 40_000_000.0 + base * 10_000.0,
            "CCC.SZ": 30_000_000.0 + base * 5_000.0,
        },
        index=idx,
    )
    benchmark = pd.Series(100.0 + base * 0.04, index=idx, name="000905.SH")
    return close, amount, benchmark


def test_cross_sectional_rank_is_date_local_and_ordered():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frame = pd.DataFrame({"A": [1.0, 30.0], "B": [2.0, 20.0], "C": [3.0, 10.0]}, index=idx)
    ranked = cross_sectional_rank(frame)

    assert ranked.loc[idx[0], "C"] > ranked.loc[idx[0], "B"] > ranked.loc[idx[0], "A"]
    assert ranked.loc[idx[1], "A"] > ranked.loc[idx[1], "B"] > ranked.loc[idx[1], "C"]
    assert ranked.max().max() <= 1.0
    assert ranked.min().min() >= -1.0


def test_v5_factor_values_do_not_change_when_future_prices_change():
    close, amount, benchmark = _sample_panels()
    cfg = V5FactorConfig()
    cutoff = close.index[150]

    before = build_v5_raw_factors(close, amount, benchmark, cfg)
    changed_close = close.copy()
    changed_amount = amount.copy()
    changed_benchmark = benchmark.copy()
    future = changed_close.index > cutoff
    changed_close.loc[future, :] *= 7.0
    changed_amount.loc[future, :] *= 11.0
    changed_benchmark.loc[future] *= 5.0
    after = build_v5_raw_factors(changed_close, changed_amount, changed_benchmark, cfg)

    for name in before:
        pd.testing.assert_series_equal(
            before[name].loc[cutoff],
            after[name].loc[cutoff],
            check_names=False,
        )


def test_relative_strength_rewards_stock_outperforming_benchmark():
    close, amount, benchmark = _sample_panels()
    factors = build_v5_raw_factors(close, amount, benchmark)
    row = factors["relative_strength_60_5"].iloc[-1]
    assert row["AAA.SZ"] > row["BBB.SH"] > row["CCC.SZ"]


def test_composite_renormalizes_available_non_missing_weights():
    idx = pd.to_datetime(["2024-01-02"])
    a = pd.DataFrame({"A": [1.0], "B": [0.5]}, index=idx)
    b = pd.DataFrame({"A": [np.nan], "B": [-0.5]}, index=idx)
    score = combine_ranked_factors({"a": a, "b": b}, {"a": 1.0, "b": 1.0})

    assert score.loc[idx[0], "A"] == 1.0
    assert score.loc[idx[0], "B"] == 0.0
