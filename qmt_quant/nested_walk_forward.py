from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .v5_oos import purged_evidence_end


@dataclass(frozen=True)
class NestedAnnualFold:
    outer_validation_year: int
    outer_train_start: pd.Timestamp
    outer_train_end: pd.Timestamp
    inner_train_start: pd.Timestamp
    inner_train_end: pd.Timestamp
    inner_validation_start: pd.Timestamp
    inner_validation_end: pd.Timestamp
    outer_validation_start: pd.Timestamp
    outer_validation_end: pd.Timestamp


@dataclass(frozen=True)
class PurgedNestedFold:
    fold: NestedAnnualFold
    inner_evidence_end: pd.Timestamp
    outer_evidence_end: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "outer_validation_year": int(self.fold.outer_validation_year),
            "outer_train_start": str(self.fold.outer_train_start.date()),
            "outer_train_end": str(self.fold.outer_train_end.date()),
            "inner_train_start": str(self.fold.inner_train_start.date()),
            "inner_train_end": str(self.fold.inner_train_end.date()),
            "inner_evidence_end": str(self.inner_evidence_end.date()),
            "inner_validation_start": str(self.fold.inner_validation_start.date()),
            "inner_validation_end": str(self.fold.inner_validation_end.date()),
            "outer_evidence_end": str(self.outer_evidence_end.date()),
            "outer_validation_start": str(self.fold.outer_validation_start.date()),
            "outer_validation_end": str(self.fold.outer_validation_end.date()),
        }


def nested_annual_folds(
    first_validation_year: int = 2021,
    last_validation_year: int = 2025,
    *,
    outer_train_years: int = 4,
    inner_validation_years: int = 1,
) -> list[NestedAnnualFold]:
    if first_validation_year > last_validation_year:
        raise ValueError("first validation year must not exceed last validation year")
    if outer_train_years < 2:
        raise ValueError("outer_train_years must be at least 2")
    if inner_validation_years <= 0 or inner_validation_years >= outer_train_years:
        raise ValueError("inner_validation_years must be positive and smaller than outer_train_years")

    folds: list[NestedAnnualFold] = []
    for year in range(int(first_validation_year), int(last_validation_year) + 1):
        outer_start_year = year - int(outer_train_years)
        inner_validation_start_year = year - int(inner_validation_years)
        folds.append(
            NestedAnnualFold(
                outer_validation_year=year,
                outer_train_start=pd.Timestamp(outer_start_year, 1, 1),
                outer_train_end=pd.Timestamp(year - 1, 12, 31),
                inner_train_start=pd.Timestamp(outer_start_year, 1, 1),
                inner_train_end=pd.Timestamp(inner_validation_start_year - 1, 12, 31),
                inner_validation_start=pd.Timestamp(inner_validation_start_year, 1, 1),
                inner_validation_end=pd.Timestamp(year - 1, 12, 31),
                outer_validation_start=pd.Timestamp(year, 1, 1),
                outer_validation_end=pd.Timestamp(year, 12, 31),
            )
        )
    return folds


def purge_nested_fold(
    fold: NestedAnnualFold,
    calendar: pd.DatetimeIndex,
    *,
    max_forward_horizon: int = 20,
) -> PurgedNestedFold:
    inner_end = purged_evidence_end(
        calendar,
        fold.inner_validation_start,
        max_forward_horizon=max_forward_horizon,
    )
    inner_end = min(inner_end, fold.inner_train_end)
    outer_end = purged_evidence_end(
        calendar,
        fold.outer_validation_start,
        max_forward_horizon=max_forward_horizon,
    )
    outer_end = min(outer_end, fold.outer_train_end)
    if inner_end < fold.inner_train_start:
        raise RuntimeError("inner training window is empty after purge")
    if outer_end < fold.outer_train_start:
        raise RuntimeError("outer training window is empty after purge")
    if inner_end >= fold.inner_validation_start:
        raise RuntimeError("inner forward labels overlap inner validation")
    if outer_end >= fold.outer_validation_start:
        raise RuntimeError("outer forward labels overlap outer validation")
    return PurgedNestedFold(fold=fold, inner_evidence_end=inner_end, outer_evidence_end=outer_end)


def choose_inner_candidate(
    candidate_metrics: Mapping[str, Mapping[str, float]],
    *,
    primary_metric: str = "sharpe",
    secondary_metric: str = "total_return",
    minimum_return: float | None = None,
) -> str:
    """Choose one already-evaluated candidate using inner-validation metrics only."""
    rows: list[tuple[float, float, str]] = []
    for name, metrics in candidate_metrics.items():
        ret = float(metrics.get("total_return", float("nan")))
        primary = float(metrics.get(primary_metric, float("nan")))
        secondary = float(metrics.get(secondary_metric, float("nan")))
        if pd.isna(primary) or pd.isna(secondary):
            continue
        if minimum_return is not None and (pd.isna(ret) or ret < float(minimum_return)):
            continue
        rows.append((primary, secondary, str(name)))
    if not rows:
        raise RuntimeError("no candidate cleared the inner-validation selection gate")
    rows.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return rows[0][2]


def assert_nested_no_leakage(rows: pd.DataFrame) -> None:
    required = {
        "inner_evidence_end",
        "inner_validation_start",
        "outer_evidence_end",
        "outer_validation_start",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"nested rows missing columns: {', '.join(missing)}")
    inner_end = pd.to_datetime(rows["inner_evidence_end"], errors="coerce")
    inner_val = pd.to_datetime(rows["inner_validation_start"], errors="coerce")
    outer_end = pd.to_datetime(rows["outer_evidence_end"], errors="coerce")
    outer_val = pd.to_datetime(rows["outer_validation_start"], errors="coerce")
    if inner_end.isna().any() or inner_val.isna().any() or outer_end.isna().any() or outer_val.isna().any():
        raise ValueError("nested fold dates contain invalid values")
    if not (inner_end < inner_val).all() or not (outer_end < outer_val).all():
        raise RuntimeError("nested walk-forward contains future-label leakage")
