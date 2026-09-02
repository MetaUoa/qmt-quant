from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from qmt_quant.backtest import calculate_metrics, run_backtest
from qmt_quant.config import CostConfig, DataConfig, StrategyConfig
from qmt_quant.qmt_data import coverage_report, load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData
from qmt_quant.reporting import save_result
from qmt_quant.research import add_neighborhood_stability, config_key, make_candidate_grid, research_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3 multi-objective parameter research; holdout is never used for selection")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--development-end", default="20221231")
    p.add_argument("--holdout-start", default="20230101")
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars")
    p.add_argument("--output", default="output/v3_research")
    p.add_argument("--profile", choices=["quick", "balanced", "deep"], default="quick")
    p.add_argument("--max-train-drawdown", type=float, default=0.50)
    p.add_argument("--min-symbol-coverage", type=float, default=0.98)
    p.add_argument("--strict-reference", action="store_true")
    return p.parse_args()


def profile_grid(profile: str, base: StrategyConfig) -> list[StrategyConfig]:
    mixes = (
        (0.20, 0.30, 0.50, 0.75),
        (0.15, 0.35, 0.50, 0.65),
        (0.30, 0.30, 0.40, 0.85),
    )
    if profile == "quick":
        return make_candidate_grid(
            base,
            top_n=(5, 8, 12),
            rebalance_days=(3, 5, 10),
            min_momentum=(0.00, 0.02),
            max_daily_vol=(0.075,),
            min_breadth=(0.0, 0.45),
            factor_mix=(mixes[0],),
        )
    if profile == "balanced":
        return make_candidate_grid(
            base,
            top_n=(5, 8, 12),
            rebalance_days=(3, 5, 10),
            min_momentum=(0.00, 0.02, 0.05),
            max_daily_vol=(0.06, 0.09),
            min_breadth=(0.0, 0.45),
            factor_mix=mixes[:2],
        )
    return make_candidate_grid(base, factor_mix=mixes)


def main() -> int:
    args = parse_args()
    cfg_data = DataConfig(
        start=args.start,
        end=args.end,
        benchmark=args.benchmark,
        reference_dir=args.reference_dir,
        bar_cache_dir=args.bar_cache_dir,
    )
    reference = ReferenceData.from_dir(cfg_data.reference_dir)
    universe = reference.codes_ever_active(cfg_data.start, cfg_data.end)
    all_codes = list(dict.fromkeys(universe + [cfg_data.benchmark]))
    cache = Path(cfg_data.bar_cache_dir) / f"{cfg_data.dividend_type}_{cfg_data.start}_{cfg_data.end}"
    bars = load_daily_bars(
        all_codes,
        cfg_data.start,
        cfg_data.end,
        dividend_type=cfg_data.dividend_type,
        batch_size=cfg_data.batch_size,
        cache_dir=cache,
    )
    raw_cache = Path(cfg_data.bar_cache_dir) / f"none_limits_{cfg_data.start}_{cfg_data.end}"
    raw = load_limit_reference_bars(
        universe,
        cfg_data.start,
        cfg_data.end,
        batch_size=cfg_data.batch_size,
        cache_dir=raw_cache,
    )
    coverage = coverage_report(universe, bars)
    ratio = float(coverage["loaded"].mean()) if len(coverage) else 0.0
    if ratio < args.min_symbol_coverage:
        raise RuntimeError(f"Historical symbol coverage {ratio:.2%} < {args.min_symbol_coverage:.2%}")

    base = StrategyConfig()
    candidates = profile_grid(args.profile, base)
    print(f"Profile={args.profile}; candidates={len(candidates)}; selection window={args.start}..{args.development_end}")
    results: dict[str, tuple[StrategyConfig, object]] = {}
    configs: dict[str, StrategyConfig] = {}
    rows: list[dict] = []
    dev_start = pd.Timestamp(args.start)
    dev_end = pd.Timestamp(args.development_end)
    hold_start = pd.Timestamp(args.holdout_start)
    hold_end = pd.Timestamp(args.end)

    for i, strategy in enumerate(candidates, 1):
        key = config_key(strategy)
        configs[key] = strategy
        print(f"[{i}/{len(candidates)}] {key}")
        result = run_backtest(
            bars,
            cfg_data.benchmark,
            strategy,
            CostConfig(),
            reference=reference,
            strict_reference=args.strict_reference,
            limit_reference_bars=raw,
        )
        results[key] = (strategy, result)
        score, diag = research_score(
            result,
            dev_start,
            dev_end,
            max_drawdown=args.max_train_drawdown,
        )
        dev_metrics = calculate_metrics(result.equity.loc[dev_start:dev_end, "equity"])
        hold_metrics = calculate_metrics(result.equity.loc[hold_start:hold_end, "equity"])
        rows.append(
            {
                "candidate": key,
                "raw_score": score,
                **{f"cfg_{k}": v for k, v in asdict(strategy).items()},
                **{f"dev_{k}": v for k, v in dev_metrics.items()},
                **{f"diag_{k}": v for k, v in diag.items()},
                # Audit only: never enters raw_score/stable_score.
                **{f"holdout_{k}": v for k, v in hold_metrics.items()},
            }
        )

    frame = add_neighborhood_stability(pd.DataFrame(rows), configs)
    finite = frame[pd.to_numeric(frame["stable_score"], errors="coerce").notna()]
    finite = finite[finite["stable_score"] != float("-inf")]
    if finite.empty:
        raise RuntimeError("No candidate passed development risk guards")
    best_key = str(finite.iloc[0]["candidate"])
    best_cfg, best_result = results[best_key]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out / "universe_coverage.csv", index=False, encoding="utf-8-sig")
    (out / "best_config.json").write_text(json.dumps(asdict(best_cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    selection = {
        "profile": args.profile,
        "candidate_count": len(candidates),
        "selection_window": [str(dev_start.date()), str(dev_end.date())],
        "holdout_window": [str(hold_start.date()), str(hold_end.date())],
        "best_candidate": best_key,
        "best_stable_score": float(finite.iloc[0]["stable_score"]),
        "holdout_used_for_selection": False,
        "symbol_coverage_ratio": ratio,
    }
    (out / "selection_audit.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    save_result(best_result, out / "best_full_result", coverage=coverage)
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
