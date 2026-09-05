from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .v5_selector import TrainingCompositeSelection, select_training_composite
from .v5_walk_forward import AnnualFold


@dataclass(frozen=True)
class PurgedFoldSelection:
    fold: AnnualFold
    evidence_end: pd.Timestamp
    selection: TrainingCompositeSelection

    def to_dict(self) -> dict:
        return {
            "validation_year": int(self.fold.validation_year),
            "train_start": str(self.fold.train_start.date()),
            "nominal_train_end": str(self.fold.train_end.date()),
            "evidence_end": str(self.evidence_end.date()),
            "validation_start": str(self.fold.validation_start.date()),
            "validation_end": str(self.fold.validation_end.date()),
            **self.selection.to_dict(),
        }


def purged_evidence_end(
    calendar: pd.DatetimeIndex,
    validation_start,
    *,
    max_forward_horizon: int,
) -> pd.Timestamp:
    """Return the last signal date whose forward label ends before validation.

    If the first validation session is at index ``i`` and a training label uses a
    ``h``-session forward return, the latest safe signal index is ``i-h-1``.
    """
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    horizon = int(max_forward_horizon)
    if horizon <= 0:
        raise ValueError("max_forward_horizon must be positive")
    validation = pd.Timestamp(validation_start).normalize()
    first_validation_i = int(calendar.searchsorted(validation, side="left"))
    safe_i = first_validation_i - horizon - 1
    if first_validation_i >= len(calendar):
        raise ValueError("validation_start is after the available calendar")
    if safe_i < 0:
        raise ValueError("insufficient history to purge forward labels")
    return pd.Timestamp(calendar[safe_i])


def select_purged_folds(
    observations: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    folds: list[AnnualFold],
    *,
    max_forward_horizon: int = 20,
) -> list[PurgedFoldSelection]:
    out: list[PurgedFoldSelection] = []
    for fold in folds:
        evidence_end = purged_evidence_end(
            calendar,
            fold.validation_start,
            max_forward_horizon=max_forward_horizon,
        )
        evidence_end = min(evidence_end, fold.train_end)
        if evidence_end < fold.train_start:
            raise RuntimeError(
                f"fold {fold.validation_year} has no training evidence after label purge"
            )
        selection = select_training_composite(
            observations,
            train_start=fold.train_start,
            train_end=evidence_end,
        )
        out.append(
            PurgedFoldSelection(
                fold=fold,
                evidence_end=evidence_end,
                selection=selection,
            )
        )
    return out


def selected_factor_union(selections: list[PurgedFoldSelection]) -> tuple[str, ...]:
    factors: set[str] = set()
    for row in selections:
        factors.update(row.selection.selected_factors)
    return tuple(sorted(factors))
