from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult


def yearly_returns(equity: pd.Series) -> pd.DataFrame:
    series = equity.dropna().sort_index()
    rows = []
    for year, group in series.groupby(series.index.year):
        if group.empty:
            continue
        start = float(group.iloc[0])
        end = float(group.iloc[-1])
        rows.append(
            {
                "year": int(year),
                "start_equity": start,
                "end_equity": end,
                "return": end / start - 1.0 if start > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_result(
    result: BacktestResult,
    output_dir: str | Path,
    coverage: pd.DataFrame | None = None,
    data_quality: dict | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(out / "equity.csv", encoding="utf-8-sig")
    result.trades.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    yearly_returns(result.equity["equity"]).to_csv(out / "yearly_returns.csv", index=False, encoding="utf-8-sig")
    if coverage is not None:
        coverage.to_csv(out / "universe_coverage.csv", index=False, encoding="utf-8-sig")
    with (out / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, ensure_ascii=False, indent=2)
    with (out / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(result.config, handle, ensure_ascii=False, indent=2)
    if data_quality is not None:
        with (out / "data_quality.json").open("w", encoding="utf-8") as handle:
            json.dump(data_quality, handle, ensure_ascii=False, indent=2)
    return out
