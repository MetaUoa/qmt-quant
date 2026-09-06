from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .reference_data import ReferenceData


@dataclass(frozen=True)
class RebalanceSelection:
    """Signal-date candidate selection before execution-day tradability checks."""

    selected: tuple[str, ...]
    blocked_st_candidates: int = 0


def select_rebalance_candidates(
    *,
    score_row: pd.Series,
    signal_date: pd.Timestamp,
    risk_on: bool,
    top_n: int,
    min_listing_sessions: int,
    reference: ReferenceData | None,
) -> RebalanceSelection:
    """Apply the existing score/PIT/ST/risk gate without execution-day checks.

    This intentionally preserves the historical backtest order of operations:
    scores are sorted first, PIT membership and exact-date ST exclusions are applied
    next, and only then does the risk gate decide whether any names are selected.
    Suspension, price-limit, T+1, fill, sizing and cash checks remain outside this
    helper because they belong to execution rather than signal-date selection.
    """
    row = score_row.dropna().sort_values(ascending=False)
    blocked_st = 0

    if reference is not None:
        member_codes = set(
            reference.filter_members(
                row.index,
                signal_date,
                int(min_listing_sessions),
            )
        )
        row = row[row.index.isin(member_codes)]
        st_codes = reference.st_codes(signal_date)
        if st_codes:
            before = len(row)
            row = row.drop(index=row.index.intersection(st_codes), errors="ignore")
            blocked_st = before - len(row)

    selected = tuple(str(code) for code in row.head(int(top_n)).index) if risk_on else ()
    return RebalanceSelection(
        selected=selected,
        blocked_st_candidates=int(blocked_st),
    )
