from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from qmt_quant.backtest import _panel
from qmt_quant.factor_diagnostics import (
    factor_observations,
    forward_return_panel,
    summarize_factor_observations,
    yearly_factor_summary,
)
from qmt_quant.factors import V5FactorConfig, iter_v5_raw_factors
from qmt_quant.qmt_data import coverage_report, load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict point-in-time full-market V5 factor diagnostics"
    )
    parser.add_argument("--data-start", default="20170101")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--benchmark", default="000905.SH")
    parser.add_argument("--reference-dir", default="data/reference")
    parser.add_argument("--bar-cache-dir", default="data/qmt_bars")
    parser.add_argument("--output", default="output/v5_factor_research")
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--horizons", default="5,20")
    parser.add_argument("--min-symbol-coverage", type=float, default=0.98)
    parser.add_argument("--min-symbols-per-date", type=int, default=50)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-price", type=float, default=3.0)
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)
    parser.add_argument("--min-listing-sessions", type=int, default=120)
    return parser.parse_args()


def _parse_horizons(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(x <= 0 for x in values):
        raise ValueError("horizons must contain positive integers")
    return values


def _coverage_or_fail(
    label: str,
    universe: list[str],
    bars: dict[str, pd.DataFrame],
    threshold: float,
) -> tuple[float, pd.DataFrame]:
    report = coverage_report(universe, bars)
    ratio = float(report["loaded"].mean()) if not report.empty else 0.0
    if ratio < float(threshold):
        raise RuntimeError(
            f"{label} symbol coverage {ratio:.4%} is below required {threshold:.4%}"
        )
    return ratio, report


def main() -> int:
    args = parse_args()
    if args.rebalance_days <= 0:
        raise ValueError("rebalance-days must be positive")
    horizons = _parse_horizons(args.horizons)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    reference = ReferenceData.from_dir(args.reference_dir)
    reference_audit = reference.audit()
    if reference_audit.st_dates < reference_audit.calendar_sessions:
        raise RuntimeError("ST snapshots are incomplete; refusing V5 factor research")
    if reference_audit.limit_dates < reference_audit.calendar_sessions:
        raise RuntimeError("daily price-limit snapshots are incomplete; refusing V5 factor research")

    universe = reference.codes_ever_active(args.start, args.end)
    if not universe:
        raise RuntimeError("point-in-time historical universe is empty")
    all_codes = list(dict.fromkeys(universe + [args.benchmark]))

    range_cache = Path(args.bar_cache_dir) / f"front_{args.data_start}_{args.end}"
    bars = load_daily_bars(
        all_codes,
        args.data_start,
        args.end,
        cache_dir=range_cache,
    )
    adjusted_coverage, adjusted_report = _coverage_or_fail(
        "adjusted", universe, bars, args.min_symbol_coverage
    )
    if args.benchmark not in bars or bars[args.benchmark].empty:
        raise RuntimeError(f"benchmark {args.benchmark} is missing")

    raw_cache = Path(args.bar_cache_dir) / f"none_limits_{args.data_start}_{args.end}"
    raw_bars = load_limit_reference_bars(
        universe,
        args.data_start,
        args.end,
        cache_dir=raw_cache,
    )
    raw_coverage, raw_report = _coverage_or_fail(
        "raw", universe, raw_bars, args.min_symbol_coverage
    )

    calendar = pd.DatetimeIndex(bars[args.benchmark].index).normalize().sort_values()
    calendar = calendar[(calendar >= pd.Timestamp(args.data_start)) & (calendar <= pd.Timestamp(args.end))]
    stock_bars = {code: frame for code, frame in bars.items() if code != args.benchmark}
    close = _panel(stock_bars, "close", calendar)
    amount = _panel(stock_bars, "amount", calendar)
    raw_close = _panel(raw_bars, "close", calendar)
    benchmark_close = bars[args.benchmark]["close"].reindex(calendar).ffill()

    factor_cfg = V5FactorConfig()
    research_start = pd.Timestamp(args.start)
    start_pos = int(calendar.searchsorted(research_start, side="left"))
    start_pos = max(start_pos, factor_cfg.warmup)
    end_pos = int(calendar.searchsorted(pd.Timestamp(args.end), side="right"))
    sample_dates = calendar[start_pos:end_pos: args.rebalance_days]
    if len(sample_dates) == 0:
        raise RuntimeError("no V5 factor research dates remain after warmup")

    missing_st_dates = [str(ts.date()) for ts in sample_dates if ts not in reference.st_dates]
    missing_limit_dates = [str(ts.date()) for ts in sample_dates if ts not in reference.limit_dates]
    if missing_st_dates or missing_limit_dates:
        raise RuntimeError(
            "strict PIT snapshots missing on research dates: "
            f"st={len(missing_st_dates)}, limit={len(missing_limit_dates)}"
        )

    avg_amount = amount.rolling(
        factor_cfg.amount_window,
        min_periods=factor_cfg.amount_window,
    ).mean().reindex(sample_dates)
    base_mask = (
        raw_close.reindex(sample_dates).ge(float(args.min_price))
        & avg_amount.ge(float(args.min_amount))
    )
    for ts in sample_dates:
        members = set(
            reference.filter_members(
                universe,
                ts,
                min_listing_sessions=args.min_listing_sessions,
            )
        )
        allowed = members.difference(reference.st_codes(ts))
        base_mask.loc[ts, :] &= base_mask.columns.isin(allowed)

    forward_samples: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        full_forward = forward_return_panel(close, horizon)
        forward_samples[horizon] = full_forward.reindex(sample_dates).where(base_mask)
        del full_forward

    observation_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    yearly_parts: list[pd.DataFrame] = []
    factor_names: list[str] = []

    for factor_name, factor_panel in iter_v5_raw_factors(
        close,
        amount,
        benchmark_close,
        factor_cfg,
    ):
        print(f"Factor: {factor_name}")
        factor_names.append(factor_name)
        sampled_factor = factor_panel.reindex(sample_dates).where(base_mask)
        del factor_panel

        for horizon in horizons:
            observations = factor_observations(
                sampled_factor,
                forward_samples[horizon],
                quantiles=args.quantiles,
                min_symbols=args.min_symbols_per_date,
            )
            if observations.empty:
                raise RuntimeError(
                    f"factor {factor_name} horizon {horizon} produced no valid observations"
                )
            observations.insert(0, "horizon", int(horizon))
            observations.insert(0, "factor", factor_name)
            observation_parts.append(observations)

            summary = summarize_factor_observations(observations)
            summary_rows.append(
                {
                    "factor": factor_name,
                    "horizon": int(horizon),
                    **summary,
                }
            )
            yearly = yearly_factor_summary(observations)
            yearly.insert(0, "horizon", int(horizon))
            yearly.insert(0, "factor", factor_name)
            yearly_parts.append(yearly)

        del sampled_factor
        gc.collect()

    observations_df = pd.concat(observation_parts, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["horizon", "mean_rank_ic"], ascending=[True, False]
    )
    yearly_df = pd.concat(yearly_parts, ignore_index=True)

    priority = (
        summary_df.groupby("factor", as_index=False)
        .agg(
            horizons=("horizon", "count"),
            mean_rank_ic=("mean_rank_ic", "mean"),
            worst_horizon_rank_ic=("mean_rank_ic", "min"),
            mean_ic_ir=("ic_ir", "mean"),
            mean_positive_ic_ratio=("positive_ic_ratio", "mean"),
            mean_top_bottom_spread=("mean_top_bottom_spread", "mean"),
            mean_positive_spread_ratio=("positive_spread_ratio", "mean"),
        )
        .sort_values(
            ["mean_rank_ic", "mean_top_bottom_spread"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    observations_df.to_csv(out / "factor_observations.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out / "factor_summary.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(out / "factor_yearly.csv", index=False, encoding="utf-8-sig")
    priority.to_csv(out / "factor_priority.csv", index=False, encoding="utf-8-sig")
    adjusted_report.to_csv(out / "adjusted_coverage.csv", index=False, encoding="utf-8-sig")
    raw_report.to_csv(out / "raw_coverage.csv", index=False, encoding="utf-8-sig")

    payload = {
        "passed": True,
        "data_start": args.data_start,
        "research_start": args.start,
        "end": args.end,
        "benchmark": args.benchmark,
        "universe_symbols": len(universe),
        "adjusted_symbol_coverage": adjusted_coverage,
        "raw_symbol_coverage": raw_coverage,
        "research_dates": int(len(sample_dates)),
        "first_research_date": str(sample_dates[0].date()),
        "last_research_date": str(sample_dates[-1].date()),
        "rebalance_days": int(args.rebalance_days),
        "horizons": horizons,
        "strict_point_in_time": True,
        "strict_st_snapshots": True,
        "strict_limit_snapshots": True,
        "min_symbol_coverage": float(args.min_symbol_coverage),
        "min_price": float(args.min_price),
        "min_amount": float(args.min_amount),
        "min_listing_sessions": int(args.min_listing_sessions),
        "factor_config": asdict(factor_cfg),
        "factors": factor_names,
        "reference_audit": asdict(reference_audit),
    }
    with (out / "research_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(priority.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
