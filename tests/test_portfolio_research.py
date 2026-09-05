import pandas as pd

from qmt_quant.portfolio_research import (
    PREDECLARED_PORTFOLIO_SPECS,
    score_weights,
    validate_portfolio_specs,
)


def test_predeclared_portfolio_grid_is_small_and_one_change_at_a_time():
    validate_portfolio_specs()
    by_name = {row.name: row for row in PREDECLARED_PORTFOLIO_SPECS}
    baseline = by_name["baseline"]
    assert (baseline.top_n, baseline.rebalance_days, baseline.weighting) == (8, 5, "equal")
    assert (by_name["top5"].rebalance_days, by_name["top5"].weighting) == (5, "equal")
    assert (by_name["top12"].rebalance_days, by_name["top12"].weighting) == (5, "equal")
    assert (by_name["rank_weighted"].top_n, by_name["rank_weighted"].rebalance_days) == (8, 5)


def test_score_weights_equal_and_rank_are_normalized_and_top_n_only():
    scores = pd.Series({"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0})
    equal = score_weights(scores, top_n=3, weighting="equal")
    rank = score_weights(scores, top_n=3, weighting="rank")
    assert abs(equal.sum() - 1.0) < 1e-12
    assert abs(rank.sum() - 1.0) < 1e-12
    assert equal["d"] == 0.0
    assert rank["d"] == 0.0
    assert rank["a"] > rank["b"] > rank["c"] > rank["d"]
