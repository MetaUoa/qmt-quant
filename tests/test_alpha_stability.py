import pandas as pd
import pytest

from qmt_quant.alpha_stability import (
    StabilityScorePolicy,
    factor_stability_scores,
    stability_reweight_selection,
)
from qmt_quant.composites import CompositeSpec
from qmt_quant.v5_selector import TrainingCompositeSelection


def _rows(factor, year, values):
    return [
        {
            "factor": factor,
            "date": f"{year}-{month:02d}-15",
            "rank_ic": value,
        }
        for month, value in enumerate(values, start=1)
    ]


def _selection():
    return TrainingCompositeSelection(
        train_start="2021-01-01",
        train_end="2022-12-31",
        spec=CompositeSpec(name="training_ic_low_redundancy", weights={"stable": 0.6, "weak": -0.4}),
        selected_factors=("stable", "weak"),
        orientations={"stable": 1, "weak": -1},
        duplicate_groups=(("stable", "other"),),
        correlation_horizon=20,
    )


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


def test_stability_reweight_preserves_c1_selection_and_ignores_post_end_rows():
    rows = []
    rows += _rows("stable", 2021, [0.03] * 6)
    rows += _rows("stable", 2022, [0.03] * 6)
    rows += _rows("weak", 2021, [-0.01] * 6)
    rows += _rows("weak", 2022, [-0.01] * 6)
    rows += _rows("stable", 2026, [-1.0] * 6)
    rows += _rows("weak", 2026, [1.0] * 6)

    result = stability_reweight_selection(
        pd.DataFrame(rows),
        _selection(),
        start="2021-01-01",
        end="2022-12-31",
    )

    assert result.selected_factors == ("stable", "weak")
    assert result.orientations == {"stable": 1, "weak": -1}
    assert result.duplicate_groups == (("stable", "other"),)
    assert result.correlation_horizon == 20
    assert result.spec.name == "training_ic_stability_weighted"
    assert result.spec.weights["stable"] > 0.0
    assert result.spec.weights["weak"] < 0.0
    assert abs(sum(abs(value) for value in result.spec.weights.values()) - 1.0) < 1e-12


def test_stability_reweight_fails_closed_when_selected_factor_lacks_evidence():
    rows = []
    rows += _rows("stable", 2021, [0.03] * 6)
    rows += _rows("stable", 2022, [0.03] * 6)
    rows += _rows("weak", 2022, [-0.01] * 6)

    with pytest.raises(RuntimeError, match="missing training stability evidence"):
        stability_reweight_selection(
            pd.DataFrame(rows),
            _selection(),
            start="2021-01-01",
            end="2022-12-31",
        )
