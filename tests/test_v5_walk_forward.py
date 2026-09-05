import pandas as pd

from qmt_quant.v5_walk_forward import annual_folds, assert_no_future_training, run_annual_walk_forward


def test_annual_folds_use_prior_three_calendar_years_only():
    folds = annual_folds(2021, 2025, train_years=3)
    assert folds[0].train_start == pd.Timestamp("2018-01-01")
    assert folds[0].train_end == pd.Timestamp("2020-12-31")
    assert folds[-1].train_start == pd.Timestamp("2022-01-01")
    assert folds[-1].validation_start == pd.Timestamp("2025-01-01")


def test_selector_never_sees_validation_year():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2018-01-01", "2025-12-31", freq="MS"),
            "value": range(96),
        }
    )

    def selector(train):
        return int(pd.to_datetime(train["date"]).dt.year.max())

    def evaluator(fitted, validation):
        validation_year = int(pd.to_datetime(validation["date"]).dt.year.min())
        assert fitted == validation_year - 1
        return {"train_last_year": fitted}

    rows = run_annual_walk_forward(frame, selector, evaluator)
    assert list(rows["validation_year"]) == [2021, 2022, 2023, 2024, 2025]
    assert_no_future_training(rows)
