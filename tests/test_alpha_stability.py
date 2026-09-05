import pandas as pd

from qmt_quant.alpha_stability import StabilityScorePolicy, factor_stability_scores


def _rows(factor, year, values):
    return [
        {
            "factor": factor,
            "date": f"{year}-{month:02d}-15",
            "rank_ic": value,
        }
        for month, value in enumerate(values, start=1)
    ]


def test_stability_scores_use_training_window_only_and_reward_consistency():
    rows = []
    rows += _rows("stable", 2021, [0.03] * 6)
    rows += _rows("stable", 2022, [0.03] * 6)
    rows += _rows("unstable", 2021, [0.08] * 6)
    rows += _rows("unstable", 2022, [-0.08] * 6)
    rows += _rows("stable", 2026, [-1.0] * 6)
    result = factor_stability_scores(
        pd.DataFrame(rows),
        start="2021-01-01",
        end="2022-12-31",
        allowed_factors=("stable", "unstable"),
    )
    by_factor = result.set_index("factor")
    assert by_factor.loc["stable", "positive_year_fraction"] == 1.0
    assert by_factor.loc["stable", "mean_rank_ic"] > 0.0
    assert by_factor.loc["stable", "icir"] <= StabilityScorePolicy().icir_cap
    assert by_factor.loc["stable", "stability_score"] > by_factor.loc["unstable", "stability_score"]


def test_stability_scores_require_enough_years():
    frame = pd.DataFrame(_rows("one_year", 2022, [0.04] * 6))
    result = factor_stability_scores(
        frame,
        start="2022-01-01",
        end="2022-12-31",
        allowed_factors=("one_year",),
    )
    assert result.empty
