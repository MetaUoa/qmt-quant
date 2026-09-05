from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioResearchSpec:
    name: str
    top_n: int
    rebalance_days: int
    weighting: str = "equal"


PREDECLARED_PORTFOLIO_SPECS = (
    PortfolioResearchSpec("baseline", top_n=8, rebalance_days=5, weighting="equal"),
    PortfolioResearchSpec("top5", top_n=5, rebalance_days=5, weighting="equal"),
    PortfolioResearchSpec("top12", top_n=12, rebalance_days=5, weighting="equal"),
    PortfolioResearchSpec("rebalance3", top_n=8, rebalance_days=3, weighting="equal"),
    PortfolioResearchSpec("rebalance10", top_n=8, rebalance_days=10, weighting="equal"),
    PortfolioResearchSpec("rank_weighted", top_n=8, rebalance_days=5, weighting="rank"),
)


def validate_portfolio_specs(specs: tuple[PortfolioResearchSpec, ...] = PREDECLARED_PORTFOLIO_SPECS) -> None:
    names = [row.name for row in specs]
    if len(names) != len(set(names)):
        raise ValueError("portfolio research spec names must be unique")
    for row in specs:
        if row.top_n <= 0 or row.rebalance_days <= 0:
            raise ValueError(f"invalid portfolio research spec: {row.name}")
        if row.weighting not in {"equal", "rank"}:
            raise ValueError(f"unsupported weighting for {row.name}: {row.weighting}")


def score_weights(scores: pd.Series, *, top_n: int, weighting: str = "equal") -> pd.Series:
    """Convert one cross-sectional score vector into normalized long-only weights.

    This is research-only and has no effect on the current backtest path until a
    nested inner-validation experiment explicitly selects a portfolio specification.
    """
    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")
    if weighting not in {"equal", "rank"}:
        raise ValueError("weighting must be 'equal' or 'rank'")
    clean = pd.to_numeric(scores, errors="coerce").dropna().sort_values(ascending=False)
    selected = clean.iloc[: int(top_n)]
    out = pd.Series(0.0, index=scores.index, dtype=float)
    if selected.empty:
        return out
    if weighting == "equal":
        weights = np.repeat(1.0 / len(selected), len(selected))
    else:
        ranks = np.arange(len(selected), 0, -1, dtype=float)
        weights = ranks / ranks.sum()
    out.loc[selected.index] = weights
    return out


validate_portfolio_specs()
