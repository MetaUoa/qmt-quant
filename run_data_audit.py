from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qmt_quant.config import DataConfig
from qmt_quant.qmt_data import download_daily_history, load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2.2 full historical QMT/PIT data audit")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars")
    p.add_argument("--output", default="output/v2_2_data_audit")
    p.add_argument("--download", action="store_true")
    p.add_argument("--min-symbol-coverage", type=float, default=0.98)
    p.add_argument("--min-session-coverage", type=float, default=0.97)
    return p.parse_args()


def count_observed_sessions(
    frame: pd.DataFrame | None,
    expected_days: pd.DatetimeIndex,
    column: str,
) -> int:
    """Count valid observations only on sessions that belong in the denominator.

    BaoStock can return placeholder rows for suspended sessions.  The audit removes
    those sessions from ``expected_days`` using ``suspend_d.parquet``; therefore the
    numerator must use the same session set or coverage can incorrectly exceed 100%.
    """
    if frame is None or frame.empty or column not in frame or len(expected_days) == 0:
        return 0
    index = pd.DatetimeIndex(frame.index).normalize()
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = pd.Series(values.notna().to_numpy(), index=index)
    valid = valid.groupby(level=0).max()
    expected = pd.DatetimeIndex(expected_days).normalize().unique()
    return int(valid.reindex(expected, fill_value=False).sum())


def main() -> int:
    args = parse_args()
    data = DataConfig(
        start=args.start,
        end=args.end,
        benchmark=args.benchmark,
        reference_dir=args.reference_dir,
        bar_cache_dir=args.bar_cache_dir,
    )
    ref = ReferenceData.from_dir(data.reference_dir)
    universe = ref.codes_ever_active(data.start, data.end)
    codes = list(dict.fromkeys(universe + [data.benchmark]))
    if args.download:
        download_daily_history(codes, data.start, data.end)

    bars = load_daily_bars(
        codes,
        data.start,
        data.end,
        dividend_type=data.dividend_type,
        batch_size=data.batch_size,
        cache_dir=Path(data.bar_cache_dir) / f"{data.dividend_type}_{data.start}_{data.end}",
    )
    raw = load_limit_reference_bars(
        universe,
        data.start,
        data.end,
        batch_size=data.batch_size,
        cache_dir=Path(data.bar_cache_dir) / f"none_limits_{data.start}_{data.end}",
    )

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    calendar = ref.calendar[(ref.calendar >= start) & (ref.calendar <= end)]
    basic = ref.stock_basic.set_index("ts_code", drop=False)
    suspend_path = Path(args.reference_dir) / "suspend_d.parquet"
    suspend_map: dict[str, set[pd.Timestamp]] = {}
    if suspend_path.exists():
        susp = pd.read_parquet(suspend_path)
        if not susp.empty and "ts_code" in susp.columns and "trade_date" in susp.columns:
            if "suspend_type" in susp.columns:
                susp = susp.loc[susp["suspend_type"].astype(str).str.upper().eq("S")]
            text = (
                susp["trade_date"]
                .astype("string")
                .str.replace(r"[^0-9]", "", regex=True)
                .str.slice(0, 8)
            )
            susp = susp.assign(
                _date=pd.to_datetime(text, format="%Y%m%d", errors="coerce")
            ).dropna(subset=["_date"])
            for code, group in susp.groupby("ts_code"):
                suspend_map[str(code)] = set(pd.DatetimeIndex(group["_date"]).normalize())
    suspension_reference_enabled = bool(suspend_path.exists())

    rows = []
    for code in universe:
        row = basic.loc[code] if code in basic.index else None
        list_date = (
            pd.Timestamp(row["list_date"])
            if row is not None and pd.notna(row["list_date"])
            else start
        )
        delist_date = (
            pd.Timestamp(row["delist_date"])
            if row is not None and pd.notna(row["delist_date"])
            else end
        )
        left, right = max(start, list_date), min(end, delist_date)
        expected_days = calendar[(calendar >= left) & (calendar <= right)]
        if code in suspend_map:
            blocked = suspend_map[code]
            expected_days = pd.DatetimeIndex([d for d in expected_days if d not in blocked])
        expected = int(len(expected_days))

        observed = count_observed_sessions(bars.get(code), expected_days, "close")
        raw_observed = count_observed_sessions(raw.get(code), expected_days, "open")
        rows.append(
            {
                "code": code,
                "expected_sessions": expected,
                "observed_sessions": observed,
                "session_coverage": observed / expected if expected else 1.0,
                "raw_observed_sessions": raw_observed,
                "raw_session_coverage": raw_observed / expected if expected else 1.0,
                "loaded": observed > 0,
                "raw_loaded": raw_observed > 0,
            }
        )

    detail = pd.DataFrame(rows)
    symbol_coverage = float(detail["loaded"].mean()) if len(detail) else 0.0
    raw_symbol_coverage = float(detail["raw_loaded"].mean()) if len(detail) else 0.0
    session_coverage = (
        float(detail["observed_sessions"].sum() / max(detail["expected_sessions"].sum(), 1))
        if len(detail)
        else 0.0
    )
    raw_session_coverage = (
        float(detail["raw_observed_sessions"].sum() / max(detail["expected_sessions"].sum(), 1))
        if len(detail)
        else 0.0
    )
    low = detail.loc[detail["session_coverage"] < args.min_session_coverage].sort_values(
        "session_coverage"
    )
    raw_low = detail.loc[
        detail["raw_session_coverage"] < args.min_session_coverage
    ].sort_values("raw_session_coverage")

    report = {
        "start": args.start,
        "end": args.end,
        "historical_symbols": len(universe),
        "symbol_coverage_ratio": symbol_coverage,
        "raw_symbol_coverage_ratio": raw_symbol_coverage,
        "session_coverage_ratio": session_coverage,
        "raw_session_coverage_ratio": raw_session_coverage,
        "low_coverage_symbols": int(len(low)),
        "low_raw_coverage_symbols": int(len(raw_low)),
        "benchmark_loaded": bool(data.benchmark in bars and not bars[data.benchmark].empty),
        "reference_audit": ref.audit().__dict__,
        "suspension_reference_enabled": suspension_reference_enabled,
        "thresholds": {
            "min_symbol_coverage": args.min_symbol_coverage,
            "min_session_coverage": args.min_session_coverage,
        },
    }
    session_gate = (
        session_coverage >= args.min_session_coverage
        and raw_session_coverage >= args.min_session_coverage
        if suspension_reference_enabled
        else True
    )
    report["warning"] = (
        None
        if suspension_reference_enabled
        else (
            "suspend_d.parquet is missing; session-coverage ratios are diagnostic only "
            "because legitimate suspension days cannot be removed."
        )
    )
    report["passed"] = bool(
        report["benchmark_loaded"]
        and symbol_coverage >= args.min_symbol_coverage
        and raw_symbol_coverage >= args.min_symbol_coverage
        and session_gate
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "symbol_session_coverage.csv", index=False, encoding="utf-8-sig")
    low.to_csv(out / "low_coverage_symbols.csv", index=False, encoding="utf-8-sig")
    raw_low.to_csv(out / "low_raw_coverage_symbols.csv", index=False, encoding="utf-8-sig")
    (out / "data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
