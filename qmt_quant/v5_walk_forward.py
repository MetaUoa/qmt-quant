from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class AnnualFold:
    validation_year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def annual_folds(
    first_validation_year: int = 2021,
    last_validation_year: int = 2025,
    *,
    train_years: int = 3,
) -> list[AnnualFold]:
    if first_validation_year > last_validation_year:
        raise ValueError("first validation year must not exceed last validation year")
    if train_years <= 0:
        raise ValueError("train_years must be positive")
    folds = []
    for year in range(int(first_validation_year), int(last_validation_year) + 1):
        folds.append(
            AnnualFold(
                validation_year=year,
                train_start=pd.Timestamp(year=year - train_years, month=1, day=1),
                train_end=pd.Timestamp(year=year - 1, month=12, day=31),
                validation_start=pd.Timestamp(year=year, month=1, day=1),
                validation_end=pd.Timestamp(year=year, month=12, day=31),
            )
        )
    return folds


def split_training_validation(frame: pd.DataFrame, fold: AnnualFold, *, date_column: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    if date_column not in frame:
        raise ValueError(f"missing date column: {date_column}")
    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column])
    train = data.loc[
        (data[date_column] >= fold.train_start) & (data[date_column] <= fold.train_end)
    ].copy()
    validation = data.loc[
        (data[date_column] >= fold.validation_start)
        & (data[date_column] <= fold.validation_end)
    ].copy()
    if not train.empty and train[date_column].max() >= fold.validation_start:
        raise RuntimeError("training data overlaps validation period")
    return train, validation


def run_annual_walk_forward(
    frame: pd.DataFrame,
    selector: Callable[[pd.DataFrame], object],
    evaluator: Callable[[object, pd.DataFrame], dict],
    *,
    first_validation_year: int = 2021,
    last_validation_year: int = 2025,
    train_years: int = 3,
    date_column: str = "date",
) -> pd.DataFrame:
    """Run reset-capital annual folds with selector access restricted to training data."""
    rows = []
    for fold in annual_folds(
        first_validation_year,
        last_validation_year,
        train_years=train_years,
    ):
        train, validation = split_training_validation(frame, fold, date_column=date_column)
        if train.empty:
            raise RuntimeError(f"fold {fold.validation_year} has no training data")
        if validation.empty:
            raise RuntimeError(f"fold {fold.validation_year} has no validation data")
        fitted = selector(train)
        metrics = evaluator(fitted, validation)
        rows.append(
            {
                "validation_year": fold.validation_year,
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "validation_start": str(fold.validation_start.date()),
                "validation_end": str(fold.validation_end.date()),
                **dict(metrics),
            }
        )
    return pd.DataFrame(rows)


def assert_no_future_training(rows: pd.DataFrame) -> None:
    required = {"train_end", "validation_start"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"walk-forward rows missing columns: {', '.join(missing)}")
    train_end = pd.to_datetime(rows["train_end"], errors="coerce")
    validation_start = pd.to_datetime(rows["validation_start"], errors="coerce")
    if train_end.isna().any() or validation_start.isna().any():
        raise ValueError("walk-forward date columns contain invalid values")
    if not (train_end < validation_start).all():
        raise RuntimeError("walk-forward contains future information in training windows")
