import pandas as pd

from qmt_quant.factor_orthogonality import (
    greedy_low_redundancy_selection,
    ic_correlation_matrix,
    panel_rank_correlation,
)


def test_panel_rank_correlation_detects_duplicate_cross_sections():
    idx = pd.date_range("2020-01-01", periods=3)
    left = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4], "C": [3, 4, 5]}, index=idx)
    right = left * 10.0
    corr = panel_rank_correlation(left, right, min_symbols=3)
    assert (corr.round(12) == 1.0).all()


def test_ic_correlation_and_greedy_selection_drop_redundant_factor():
    rows = []
    for date, a, b, c in [
        ("2020-01-01", 0.10, 0.10, -0.02),
        ("2020-01-08", 0.20, 0.20, 0.01),
        ("2020-01-15", 0.30, 0.30, 0.03),
        ("2020-01-22", 0.40, 0.40, -0.01),
    ]:
        for factor, value in [("a", a), ("b", b), ("c", c)]:
            rows.append({"factor": factor, "date": date, "horizon": 5, "rank_ic": value})
    observations = pd.DataFrame(rows)
    corr = ic_correlation_matrix(observations, horizon=5)
    priority = pd.DataFrame(
        {"factor": ["a", "b", "c"], "mean_rank_ic": [0.30, 0.29, 0.05]}
    )
    selected = greedy_low_redundancy_selection(priority, corr, max_abs_correlation=0.8)
    accepted = selected.loc[selected["accepted"], "factor"].tolist()
    assert "a" in accepted
    assert "b" not in accepted
    assert "c" in accepted
