from __future__ import annotations

import argparse
import json
from pathlib import Path

from qmt_quant.config import CostConfig, DataConfig, StrategyConfig
from qmt_quant.qmt_data import coverage_report, load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData
from qmt_quant.stress import monte_carlo_daily_returns, run_stress_suite, stress_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V4.5 robustness/stress test suite")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars")
    p.add_argument("--strategy-config", default="output/v3_research/best_config.json")
    p.add_argument("--output", default="output/v4_5_stress")
    p.add_argument("--strict-reference", action="store_true")
    p.add_argument("--min-symbol-coverage", type=float, default=0.98)
    p.add_argument("--monte-carlo", type=int, default=1000)
    return p.parse_args()


def load_strategy(path: str) -> StrategyConfig:
    p = Path(path)
    if not p.exists():
        print(f"WARNING: {p} not found; using default StrategyConfig")
        return StrategyConfig()
    payload = json.loads(p.read_text(encoding="utf-8"))
    allowed = StrategyConfig.__dataclass_fields__.keys()
    return StrategyConfig(**{k: v for k, v in payload.items() if k in allowed})


def main() -> int:
    args = parse_args()
    data = DataConfig(
        start=args.start,
        end=args.end,
        benchmark=args.benchmark,
        reference_dir=args.reference_dir,
        bar_cache_dir=args.bar_cache_dir,
    )
    reference = ReferenceData.from_dir(data.reference_dir)
    universe = reference.codes_ever_active(data.start, data.end)
    all_codes = list(dict.fromkeys(universe + [data.benchmark]))
    bars = load_daily_bars(
        all_codes,
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
    coverage = coverage_report(universe, bars)
    ratio = float(coverage["loaded"].mean()) if len(coverage) else 0.0
    if ratio < args.min_symbol_coverage:
        raise RuntimeError(f"Historical symbol coverage {ratio:.2%} < {args.min_symbol_coverage:.2%}")

    strategy = load_strategy(args.strategy_config)
    frame, results = run_stress_suite(
        bars,
        data.benchmark,
        strategy,
        CostConfig(),
        reference=reference,
        strict_reference=args.strict_reference,
        limit_reference_bars=raw,
    )
    summary = stress_summary(frame)
    base = results["base"]
    mc = monte_carlo_daily_returns(base.equity["equity"], simulations=args.monte_carlo)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "stress_scenarios.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out / "universe_coverage.csv", index=False, encoding="utf-8-sig")
    (out / "stress_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "monte_carlo.json").write_text(json.dumps(mc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stress": summary, "monte_carlo": mc}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
