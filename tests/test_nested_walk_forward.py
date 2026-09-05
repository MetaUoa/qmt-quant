import pandas as pd
import pytest

from qmt_quant.nested_walk_forward import (
    assert_nested_no_leakage,
    choose_inner_candidate,
    nested_annual_folds,
    purge_nested_fold,
)


def test_nested_folds_reserve_inner_validation_before_outer_year():
    folds = nested_annual_folds(2021, 2022, outer_train_years=4, inner_validation_years=1)
    first = folds[0]
    assert first.outer_train_start == pd.Timestamp("2017-01-01")
    assert first.inner_train_end == pd.Timestamp("2019-12-31")
    assert first.inner_validation_start == pd.Timestamp("2020-01-01")
    assert first.outer_validation_start == pd.Timestamp("2021-01-01")


def test_nested_purge_keeps_forward_labels_out_of_both_validation_windows():
    calendar = pd.bdate_range("2017-01-02", "2022-12-30")
    fold = nested_annual_folds(2021, 2021)[0]
    purged = purge_nested_fold(fold, calendar, max_forward_horizon=20)
    assert purged.inner_evidence_end < fold.inner_validation_start
    assert purged.outer_evidence_end < fold.outer_validation_start
    frame = pd.DataFrame([purged.to_dict()])
    assert_nested_no_leakage(frame)


def test_inner_candidate_selection_uses_predeclared_metrics():
    chosen = choose_inner_candidate(
        {
            "raw": {"sharpe": 0.2, "total_return": 0.1},
            "neutral": {"sharpe": 0.6, "total_return": 0.05},
        }
    )
    assert chosen == "neutral"
    with pytest.raises(RuntimeError):
        choose_inner_candidate({"bad": {"sharpe": float("nan"), "total_return": -1.0}})
