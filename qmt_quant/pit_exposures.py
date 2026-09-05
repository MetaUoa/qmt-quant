from __future__ import annotations

import numpy as np
import pandas as pd

from .free_data import baostock_to_ts_code


EXPOSURE_SOURCE = "baostock_turnover_implied_float_cap"


def turnover_implied_float_market_cap(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a PIT free-float market-cap proxy from same-day BaoStock fields.

    BaoStock ``turn`` is a percentage turnover rate.  With volume in shares,
    free-float shares are inferred as ``volume / (turn / 100)``.  Suspended or
    zero-turnover rows remain missing rather than being back/forward filled here.
    """
    required = {"date", "close", "volume", "turn"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"exposure history missing columns: {', '.join(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("close", "volume", "turn"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    valid = (
        out["date"].notna()
        & out["close"].gt(0.0)
        & out["volume"].gt(0.0)
        & out["turn"].gt(0.0)
    )
    out["float_shares_implied"] = np.nan
    out.loc[valid, "float_shares_implied"] = (
        out.loc[valid, "volume"] / (out.loc[valid, "turn"] / 100.0)
    )
    out["float_market_cap"] = out["close"] * out["float_shares_implied"]
    out["log_float_market_cap"] = np.log(out["float_market_cap"].where(out["float_market_cap"].gt(0.0)))
    out["exposure_source"] = EXPOSURE_SOURCE
    return out[
        [
            "date",
            "float_shares_implied",
            "float_market_cap",
            "log_float_market_cap",
            "exposure_source",
        ]
    ].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def monthly_first_open_dates(trade_calendar: pd.DataFrame) -> pd.DatetimeIndex:
    if "cal_date" not in trade_calendar or "is_open" not in trade_calendar:
        raise ValueError("trade calendar must contain cal_date and is_open")
    frame = trade_calendar.loc[pd.to_numeric(trade_calendar["is_open"], errors="coerce").eq(1)].copy()
    frame["date"] = pd.to_datetime(frame["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    first = frame.groupby(frame["date"].dt.to_period("M"), sort=True)["date"].min()
    return pd.DatetimeIndex(first.to_list())


def normalize_industry_snapshot(frame: pd.DataFrame, *, asof_date) -> pd.DataFrame:
    if "code" not in frame or "industry" not in frame:
        raise ValueError("industry snapshot must contain code and industry")
    out = frame.copy()
    out = out.loc[out["code"].astype(str).str.lower().str.match(r"^(sh|sz)\.\d{6}$")].copy()
    out["ts_code"] = out["code"].map(baostock_to_ts_code)
    out["industry"] = out["industry"].astype("string").str.strip().replace("", pd.NA)
    out["asof_date"] = pd.Timestamp(asof_date).normalize()
    classification = out.get("industryClassification", pd.Series(pd.NA, index=out.index, dtype="string"))
    out["industry_classification"] = classification.astype("string")
    return out[
        ["asof_date", "ts_code", "industry", "industry_classification"]
    ].drop_duplicates(["asof_date", "ts_code"], keep="last").reset_index(drop=True)


def asof_industry_panel(
    snapshots: pd.DataFrame,
    dates: pd.DatetimeIndex,
    codes: list[str],
) -> pd.DataFrame:
    """Forward-fill only from snapshots already known on or before each date."""
    required = {"asof_date", "ts_code", "industry"}
    missing = sorted(required.difference(snapshots.columns))
    if missing:
        raise ValueError(f"industry snapshots missing columns: {', '.join(missing)}")
    snap = snapshots.copy()
    snap["asof_date"] = pd.to_datetime(snap["asof_date"], errors="coerce")
    snap = snap.dropna(subset=["asof_date", "ts_code"])
    wide = snap.pivot_table(index="asof_date", columns="ts_code", values="industry", aggfunc="last")
    target = wide.index.union(pd.DatetimeIndex(dates)).sort_values()
    wide = wide.reindex(target).ffill().reindex(pd.DatetimeIndex(dates))
    return wide.reindex(columns=list(codes))
