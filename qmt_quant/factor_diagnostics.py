from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def forward_return_panel(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from factor date t to t+h, used only as an ex-post research label."""
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return close.shift(-horizon).div(close).sub(1.0)


def rank_ic(factor: pd.Series, forward_return: pd.Series, min_symbols: int = 20) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(factor, errors="coerce").rename("factor"),
            pd.to_numeric(forward_return, errors="coerce").rename("forward"),
        ],
        axis=1,
    ).dropna()
    if len(pair) < int(min_symbols):
        return float("nan")
    factor_rank = pair["factor"].rank(method="average")
    return_rank = pair["forward"].rank(method="average")
    value = factor_rank.corr(return_rank)
    return float(value) if pd.notna(value) else float("nan")


def _quantile_means(
    factor: pd.Series,
    forward_return: pd.Series,
    quantiles: int,
    min_symbols: int,
) -> dict[str, float]:
    pair = pd.concat(
        [
            pd.to_numeric(factor, errors="coerce").rename("factor"),
            pd.to_numeric(forward_return, errors="coerce").rename("forward"),
        ],
        axis=1,
    ).dropna()
    quantiles = int(quantiles)
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    if len(pair) < max(int(min_symbols), quantiles * 2):
        return {f"q{i}_return": float("nan") for i in range(1, quantiles + 1)}

    pct_rank = pair["factor"].rank(method="first", pct=True)
    bucket = np.ceil(pct_rank * quantiles).clip(1, quantiles).astype(int)
    grouped = pair["forward"].groupby(bucket).mean()
    return {
        f"q{i}_return": float(grouped.get(i, np.nan))
        for i in range(1, quantiles + 1)
    }


def factor_observations(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    *,
    dates: Iterable[pd.Timestamp] | None = None,
    quantiles: int = 5,
    min_symbols: int = 20,
) -> pd.DataFrame:
    """Calculate date-local IC and quantile spread diagnostics for one factor."""
    common_dates = factor.index.intersection(forward_return.index)
    if dates is not None:
        requested = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
        common_dates = common_dates.intersection(requested)

    rows: list[dict] = []
    for ts in common_dates.sort_values():
        f = factor.loc[ts]
        r = forward_return.loc[ts]
        pair = pd.concat(
            [
                pd.to_numeric(f, errors="coerce").rename("factor"),
                pd.to_numeric(r, errors="coerce").rename("forward"),
            ],
            axis=1,
        ).dropna()
        if len(pair) < int(min_symbols):
            continue
        ic = rank_ic(pair["factor"], pair["forward"], min_symbols=min_symbols)
        q = _quantile_means(
            pair["factor"],
            pair["forward"],
            quantiles=quantiles,
            min_symbols=min_symbols,
        )
        top = q.get(f"q{int(quantiles)}_return", np.nan)
        bottom = q.get("q1_return", np.nan)
        rows.append(
            {
                "date": pd.Timestamp(ts).normalize(),
                "symbols": int(len(pair)),
                "rank_ic": ic,
                "top_bottom_spread": float(top - bottom)
                if pd.notna(top) and pd.notna(bottom)
                else float("nan"),
                **q,
            }
        )
    return pd.DataFrame(rows)


def summarize_factor_observations(observations: pd.DataFrame) -> dict:
    if observations.empty:
        return {
            "dates": 0,
            "mean_rank_ic": float("nan"),
            "median_rank_ic": float("nan"),
            "rank_ic_std": float("nan"),
            "ic_ir": float("nan"),
            "positive_ic_ratio": float("nan"),
            "mean_top_bottom_spread": float("nan"),
            "positive_spread_ratio": float("nan"),
        }
    ic = pd.to_numeric(observations["rank_ic"], errors="coerce").dropna()
    spread = pd.to_numeric(observations["top_bottom_spread"], errors="coerce").dropna()
    ic_std = float(ic.std(ddof=0)) if len(ic) else float("nan")
    mean_ic = float(ic.mean()) if len(ic) else float("nan")
    return {
        "dates": int(len(observations)),
        "mean_rank_ic": mean_ic,
        "median_rank_ic": float(ic.median()) if len(ic) else float("nan"),
        "rank_ic_std": ic_std,
        "ic_ir": float(mean_ic / ic_std) if np.isfinite(ic_std) and ic_std > 0 else float("nan"),
        "positive_ic_ratio": float((ic > 0.0).mean()) if len(ic) else float("nan"),
        "mean_top_bottom_spread": float(spread.mean()) if len(spread) else float("nan"),
        "positive_spread_ratio": float((spread > 0.0).mean()) if len(spread) else float("nan"),
    }


def yearly_factor_summary(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "dates",
                "mean_rank_ic",
                "positive_ic_ratio",
                "mean_top_bottom_spread",
                "positive_spread_ratio",
            ]
        )
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame["year"] = frame["date"].dt.year
    rows: list[dict] = []
    for year, group in frame.groupby("year", sort=True):
        summary = summarize_factor_observations(group)
        rows.append(
            {
                "year": int(year),
                "dates": summary["dates"],
                "mean_rank_ic": summary["mean_rank_ic"],
                "positive_ic_ratio": summary["positive_ic_ratio"],
                "mean_top_bottom_spread": summary["mean_top_bottom_spread"],
                "positive_spread_ratio": summary["positive_spread_ratio"],
            }
        )
    return pd.DataFrame(rows)
