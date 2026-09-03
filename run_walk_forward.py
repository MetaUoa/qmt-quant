from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from qmt_quant.backtest import calculate_metrics
from qmt_quant.config import CostConfig, DataConfig, StrategyConfig
from qmt_quant.qmt_data import coverage_report, load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData
from qmt_quant.research import research_score
from qmt_quant.windowed import run_window_backtest


def _ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent fold-by-fold walk-forward validation for QMT Quant"
    )
    parser.add_argument("--data-start", default="20170101", help="Warmup data may begin before research start")
    parser.add_argument("--start", default="20180101", help="First research/training date")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--benchmark", default="000905.SH")
    parser.add_argument("--reference-dir", default="data/reference")
    parser.add_argument("--bar-cache-dir", default="data/qmt_bars")
    parser.add_argument("--output", default="output/walk_forward")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--top-n-grid", default="5,8,12")
    parser.add_argument("--rebalance-grid", default="3,5,10")
    parser.add_argument("--min-momentum-grid", default="0.00,0.02")
    parser.add_argument("--max-daily-vol-grid", default="0.075")
    parser.add_argument("--min-breadth-grid", default="0.00,0.45")
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)
    parser.add_argument("--min-listing-sessions", type=int, default=120)
    parser.add_argument("--max-train-drawdown", type=float, default=0.50)
    parser.add_argument("--strict-reference", action="store_true")
    return parser.parse_args()


def _train_score(metrics: dict, max_dd: float) -> float:
    if not metrics:
        return float("-inf")
    drawdown = abs(float(metrics.get("max_drawdown", -1.0)))
    if drawdown > max_dd:
        return float("-inf")
    return (
        0.55 * float(metrics.get("calmar", 0.0))
        + 0.30 * float(metrics.get("sharpe", 0.0))
        + 0.15 * float(metrics.get("cagr", 0.0))
    )


def candidate_key(cfg: StrategyConfig) -> str:
    return (
        f"top{cfg.top_n}_reb{cfg.rebalance_days}_mom{cfg.min_momentum:.3f}"
        f"_vol{cfg.max_daily_vol:.3f}_breadth{cfg.min_breadth:.2f}"
    )


def main() -> None:
    args = parse_args()
    if pd.Timestamp(args.data_start) > pd.Timestamp(args.start):
        raise ValueError("data-start must not be after research start")

    data_cfg = DataConfig(
        start=args.data_start,
        end=args.end,
        benchmark=args.benchmark,
        reference_dir=args.reference_dir,
        bar_cache_dir=args.bar_cache_dir,
    )
    reference = ReferenceData.from_dir(data_cfg.reference_dir)
    universe = reference.codes_ever_active(args.start, args.end)
    all_codes = list(dict.fromkeys(universe + [data_cfg.benchmark]))
    range_cache = Path(data_cfg.bar_cache_dir) / f"{data_cfg.dividend_type}_{data_cfg.start}_{data_cfg.end}"
    bars = load_daily_bars(
        all_codes,
        data_cfg.start,
        data_cfg.end,
        dividend_type=data_cfg.dividend_type,
        batch_size=data_cfg.batch_size,
        cache_dir=range_cache,
    )
    if data_cfg.benchmark not in bars:
        raise RuntimeError(f"Benchmark {data_cfg.benchmark} is missing")
    raw_cache = Path(data_cfg.bar_cache_dir) / f"none_limits_{data_cfg.start}_{data_cfg.end}"
    limit_reference_bars = load_limit_reference_bars(
        universe,
        data_cfg.start,
        data_cfg.end,
        batch_size=data_cfg.batch_size,
        cache_dir=raw_cache,
    )

    cov = coverage_report(universe, bars)
    ratio = float(cov["loaded"].mean()) if not cov.empty else 0.0
    print(f"Loaded historical symbol coverage: {ratio:.2%}")

    base = StrategyConfig(
        min_amount=args.min_amount,
        min_listing_sessions=args.min_listing_sessions,
    )
    candidates: list[StrategyConfig] = []
    for top_n, rebalance_days, min_momentum, max_daily_vol, min_breadth in itertools.product(
        _ints(args.top_n_grid),
        _ints(args.rebalance_grid),
        _floats(args.min_momentum_grid),
        _floats(args.max_daily_vol_grid),
        _floats(args.min_breadth_grid),
    ):
        candidates.append(
            replace(
                base,
                top_n=top_n,
                rebalance_days=rebalance_days,
                min_momentum=min_momentum,
                max_daily_vol=max_daily_vol,
                min_breadth=min_breadth,
            )
        )

    start_year = pd.Timestamp(args.start).year
    end_year = pd.Timestamp(args.end).year
    first_validation = start_year + args.train_years
    fold_rows: list[dict] = []
    candidate_rows: list[dict] = []
    stitched_parts: list[pd.Series] = []
    chained_capital = 1.0

    for validation_year in range(first_validation, end_year + 1):
        train_start = pd.Timestamp(year=validation_year - args.train_years, month=1, day=1)
        train_end = pd.Timestamp(year=validation_year - 1, month=12, day=31)
        val_start = pd.Timestamp(year=validation_year, month=1, day=1)
        val_end = min(pd.Timestamp(year=validation_year, month=12, day=31), pd.Timestamp(args.end))

        print(
            f"Fold {validation_year}: train {train_start.date()}..{train_end.date()}, "
            f"validate {val_start.date()}..{val_end.date()}"
        )
        best = None
        for idx, cfg in enumerate(candidates, 1):
            key = candidate_key(cfg)
            train_result = run_window_backtest(
                bars,
                data_cfg.benchmark,
                cfg,
                CostConfig(),
                trade_start=train_start,
                trade_end=train_end,
                reference=reference,
                strict_reference=args.strict_reference,
                limit_reference_bars=limit_reference_bars,
            )
            score, diagnostics = research_score(
                train_result,
                max_drawdown=args.max_train_drawdown,
            )
            candidate_rows.append(
                {
                    "validation_year": validation_year,
                    "candidate": key,
                    "train_score": score,
                    **asdict(cfg),
                    **{
                        f"train_{k}": v
                        for k, v in train_result.metrics.items()
                        if k in {"cagr", "multiple", "max_drawdown", "sharpe", "calmar", "trade_count"}
                    },
                    **diagnostics,
                    "warmup_truncated": train_result.metrics.get("warmup_truncated", False),
                }
            )
            if idx == 1 or idx % 12 == 0 or idx == len(candidates):
                print(f"  candidate {idx}/{len(candidates)}")
            if np.isfinite(score) and (best is None or score > best[0]):
                best = (score, key, cfg, train_result)

        if best is None:
            raise RuntimeError(
                f"No candidate passed training drawdown guard for validation year {validation_year}"
            )
        score, key, cfg, train_result = best

        validation = run_window_backtest(
            bars,
            data_cfg.benchmark,
            cfg,
            CostConfig(),
            trade_start=val_start,
            trade_end=val_end,
            reference=reference,
            strict_reference=args.strict_reference,
            limit_reference_bars=limit_reference_bars,
        )
        val_metrics = validation.metrics
        if not val_metrics:
            continue
        val_eq = validation.equity["equity"].dropna()
        normalized = val_eq / float(val_eq.iloc[0]) * chained_capital
        if stitched_parts and not normalized.empty:
            normalized = normalized.iloc[1:]
        stitched_parts.append(normalized)
        chained_capital *= float(val_metrics.get("multiple", 1.0))

        fold_rows.append(
            {
                "validation_year": validation_year,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "validation_start": str(val_start.date()),
                "validation_end": str(val_end.date()),
                "candidate": key,
                "train_score": score,
                "train_cagr": train_result.metrics.get("cagr"),
                "train_max_drawdown": train_result.metrics.get("max_drawdown"),
                "train_sharpe": train_result.metrics.get("sharpe"),
                "train_warmup_truncated": train_result.metrics.get("warmup_truncated", False),
                "validation_return": val_metrics.get("total_return"),
                "validation_multiple": val_metrics.get("multiple"),
                "validation_max_drawdown": val_metrics.get("max_drawdown"),
                "validation_sharpe": val_metrics.get("sharpe"),
                "validation_warmup_truncated": val_metrics.get("warmup_truncated", False),
                "top_n": cfg.top_n,
                "rebalance_days": cfg.rebalance_days,
                "min_momentum": cfg.min_momentum,
                "max_daily_vol": cfg.max_daily_vol,
                "min_breadth": cfg.min_breadth,
            }
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_rows).to_csv(
        out / "candidate_fold_training.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_rows).to_csv(
        out / "walk_forward_folds.csv", index=False, encoding="utf-8-sig"
    )
    cov.to_csv(out / "universe_coverage.csv", index=False, encoding="utf-8-sig")

    if candidate_rows:
        candidates_df = pd.DataFrame(candidate_rows)
        summary = (
            candidates_df.groupby("candidate", as_index=False)
            .agg(
                folds=("validation_year", "count"),
                mean_train_score=("train_score", "mean"),
                median_train_score=("train_score", "median"),
                mean_train_cagr=("train_cagr", "mean"),
                worst_train_drawdown=("train_max_drawdown", "min"),
            )
            .sort_values("mean_train_score", ascending=False)
        )
        summary.to_csv(out / "candidate_summary.csv", index=False, encoding="utf-8-sig")

    if stitched_parts:
        stitched = pd.concat(stitched_parts).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="last")]
        stitched.to_frame("oos_equity").to_csv(out / "oos_equity.csv", encoding="utf-8-sig")
        oos_metrics = calculate_metrics(stitched)
    else:
        oos_metrics = {}
    fold_df = pd.DataFrame(fold_rows)
    oos_metrics.update(
        {
            "symbol_coverage_ratio": ratio,
            "fold_count": len(fold_rows),
            "positive_oos_folds": int(
                (fold_df.get("validation_return", pd.Series(dtype=float)) > 0).sum()
            ),
            "method": "independent_reset_folds",
            "method_note": (
                "Every training candidate is rerun with reset capital and only pre-end data. "
                "The selected configuration is then rerun in a separate validation window with reset capital. "
                "Pre-period bars are used only as backward-looking feature warmup."
            ),
        }
    )
    with (out / "walk_forward_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(oos_metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(oos_metrics, ensure_ascii=False, indent=2))
    print(f"Saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
