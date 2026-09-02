from __future__ import annotations

import argparse
import json
from pathlib import Path

from qmt_quant.backtest import run_backtest
from qmt_quant.config import CostConfig, DataConfig, StrategyConfig
from qmt_quant.qmt_data import (
    coverage_report,
    download_daily_history,
    load_daily_bars,
    load_limit_reference_bars,
    read_universe_file,
)
from qmt_quant.reference_data import ReferenceData
from qmt_quant.reporting import save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QMT V2 point-in-time A-share momentum rotation backtest")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--benchmark", default="000905.SH")
    parser.add_argument("--reference-dir", default="data/reference")
    parser.add_argument("--bar-cache-dir", default="data/qmt_bars")
    parser.add_argument("--universe-file", default="", help="Optional restriction list; PIT membership is still enforced")
    parser.add_argument("--max-stocks", type=int, default=0, help="Smoke-test only; sorted-prefix sample")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output", default="output/v2_baseline")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)
    parser.add_argument("--min-listing-sessions", type=int, default=120)
    parser.add_argument("--strict-reference", action="store_true")
    parser.add_argument("--min-symbol-coverage", type=float, default=0.98)
    parser.add_argument("--no-reference", action="store_true", help="Unsafe compatibility mode; reintroduces survivorship risk")
    parser.add_argument("--skip-raw-limit-reference", action="store_true", help="Use adjusted-bar ratio fallback instead of raw QMT open/preClose")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig(
        start=args.start,
        end=args.end,
        benchmark=args.benchmark,
        reference_dir=args.reference_dir,
        bar_cache_dir=args.bar_cache_dir,
    )
    strategy_cfg = StrategyConfig(
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        min_amount=args.min_amount,
        min_listing_sessions=args.min_listing_sessions,
    )
    cost_cfg = CostConfig()

    reference = None if args.no_reference else ReferenceData.from_dir(data_cfg.reference_dir)
    if args.universe_file:
        universe = read_universe_file(args.universe_file)
        if reference is not None:
            ever = set(reference.codes_ever_active(data_cfg.start, data_cfg.end))
            universe = [code for code in universe if code in ever]
    elif reference is not None:
        universe = reference.codes_ever_active(data_cfg.start, data_cfg.end)
    else:
        from qmt_quant.qmt_data import get_sector_universe

        universe = get_sector_universe(data_cfg.sector)

    if args.max_stocks > 0:
        universe = universe[: args.max_stocks]
    all_codes = list(dict.fromkeys(universe + [data_cfg.benchmark]))
    print(f"Historical candidate symbols: {len(universe)}")

    if args.download:
        print("Downloading/refreshing QMT local daily history...")
        download_daily_history(all_codes, data_cfg.start, data_cfg.end)

    range_cache = Path(data_cfg.bar_cache_dir) / f"{data_cfg.dividend_type}_{data_cfg.start}_{data_cfg.end}"
    print(f"Loading QMT local daily bars (Parquet mirror: {range_cache})...")
    bars = load_daily_bars(
        all_codes,
        data_cfg.start,
        data_cfg.end,
        dividend_type=data_cfg.dividend_type,
        batch_size=data_cfg.batch_size,
        cache_dir=range_cache,
        refresh_cache=args.refresh_cache,
    )
    if data_cfg.benchmark not in bars:
        raise RuntimeError(f"Benchmark {data_cfg.benchmark} has no local QMT daily data.")

    limit_reference_bars = None
    if reference is not None and not args.skip_raw_limit_reference:
        raw_cache = Path(data_cfg.bar_cache_dir) / f"none_limits_{data_cfg.start}_{data_cfg.end}"
        print(f"Loading unadjusted QMT open/preClose for exact limit checks (cache: {raw_cache})...")
        limit_reference_bars = load_limit_reference_bars(
            universe,
            data_cfg.start,
            data_cfg.end,
            batch_size=data_cfg.batch_size,
            cache_dir=raw_cache,
            refresh_cache=args.refresh_cache,
        )

    coverage = coverage_report(universe, bars)
    loaded = int(coverage["loaded"].sum()) if not coverage.empty else 0
    ratio = loaded / max(len(universe), 1)
    raw_loaded = (
        sum(1 for code in universe if limit_reference_bars is not None and code in limit_reference_bars and not limit_reference_bars[code].empty)
        if limit_reference_bars is not None
        else 0
    )
    raw_ratio = raw_loaded / max(len(universe), 1) if limit_reference_bars is not None else None
    quality = {
        "requested_symbols": len(universe),
        "loaded_symbols": loaded,
        "symbol_coverage_ratio": ratio,
        "benchmark_loaded": True,
        "raw_limit_reference_symbols": raw_loaded if limit_reference_bars is not None else None,
        "raw_limit_reference_coverage_ratio": raw_ratio,
        "reference_enabled": reference is not None,
        "reference_audit": reference.audit().__dict__ if reference is not None else None,
        "warning": None,
    }
    if ratio < args.min_symbol_coverage:
        quality["warning"] = (
            f"Only {ratio:.2%} of the historical candidate symbols have QMT bars. "
            "Missing delisted/old symbols can reintroduce survivorship bias."
        )
        print("WARNING:", quality["warning"])
        if args.strict_reference:
            raise RuntimeError(quality["warning"])
    if args.strict_reference and raw_ratio is not None and raw_ratio < args.min_symbol_coverage:
        raise RuntimeError(
            f"Only {raw_ratio:.2%} of historical symbols have raw QMT open/preClose for exact limit checks."
        )

    result = run_backtest(
        bars,
        data_cfg.benchmark,
        strategy_cfg,
        cost_cfg,
        reference=reference,
        strict_reference=args.strict_reference,
        limit_reference_bars=limit_reference_bars,
    )
    result.metrics["symbol_coverage_ratio"] = ratio
    out = save_result(result, args.output, coverage=coverage, data_quality=quality)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    print(f"Saved to: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
