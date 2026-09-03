from __future__ import annotations

import argparse
import json
from pathlib import Path

from qmt_quant.config import CostConfig, DataConfig, StrategyConfig
from qmt_quant.qmt_data import coverage_report, load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData
from qmt_quant.reporting import save_result
from qmt_quant.windowed import run_window_backtest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict baseline with pre-period warmup and reset capital")
    p.add_argument("--data-start", default="20170101")
    p.add_argument("--trade-start", default="20180101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars")
    p.add_argument("--output", default="output/v4_1_full_market_baseline")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--rebalance-days", type=int, default=5)
    p.add_argument("--min-symbol-coverage", type=float, default=0.98)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = DataConfig(
        start=args.data_start,
        end=args.end,
        benchmark=args.benchmark,
        reference_dir=args.reference_dir,
        bar_cache_dir=args.bar_cache_dir,
    )
    reference = ReferenceData.from_dir(data.reference_dir)
    universe = reference.codes_ever_active(args.trade_start, args.end)
    all_codes = list(dict.fromkeys(universe + [data.benchmark]))
    front_root = Path(data.bar_cache_dir) / f"front_{data.start}_{data.end}"
    raw_root = Path(data.bar_cache_dir) / f"none_limits_{data.start}_{data.end}"
    bars = load_daily_bars(
        all_codes,
        data.start,
        data.end,
        dividend_type=data.dividend_type,
        batch_size=data.batch_size,
        cache_dir=front_root,
    )
    raw = load_limit_reference_bars(
        universe,
        data.start,
        data.end,
        batch_size=data.batch_size,
        cache_dir=raw_root,
    )
    coverage = coverage_report(universe, bars)
    ratio = float(coverage["loaded"].mean()) if not coverage.empty else 0.0
    raw_ratio = sum(1 for code in universe if code in raw and not raw[code].empty) / max(len(universe), 1)
    if ratio < args.min_symbol_coverage or raw_ratio < args.min_symbol_coverage:
        raise RuntimeError(
            f"Coverage gate failed: adjusted={ratio:.2%}, raw={raw_ratio:.2%}"
        )

    strategy = StrategyConfig(top_n=args.top_n, rebalance_days=args.rebalance_days)
    result = run_window_backtest(
        bars,
        data.benchmark,
        strategy,
        CostConfig(),
        trade_start=args.trade_start,
        trade_end=args.end,
        reference=reference,
        strict_reference=True,
        limit_reference_bars=raw,
    )
    result.metrics["symbol_coverage_ratio"] = ratio
    quality = {
        "requested_symbols": len(universe),
        "symbol_coverage_ratio": ratio,
        "raw_limit_reference_coverage_ratio": raw_ratio,
        "reference_audit": reference.audit().__dict__,
        "strict_reference": True,
    }
    save_result(result, args.output, coverage=coverage, data_quality=quality)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
