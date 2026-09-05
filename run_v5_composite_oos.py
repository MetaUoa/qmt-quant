from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qmt_quant.backtest import _panel, calculate_metrics
from qmt_quant.composites import apply_composite
from qmt_quant.config import CostConfig, StrategyConfig
from qmt_quant.factors import V5FactorConfig, iter_v5_raw_factors, normalize_factor
from qmt_quant.qmt_data import coverage_report, load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData
from qmt_quant.v5_gates import evaluate_v5_gates
from qmt_quant.v5_oos import select_purged_folds, selected_factor_union
from qmt_quant.v5_walk_forward import annual_folds, assert_no_future_training
from qmt_quant.windowed import context_start_for_window, run_window_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict training-only V5 composite 2021-2025 walk-forward/OOS"
    )
    parser.add_argument("--factor-observations", required=True)
    parser.add_argument("--data-start", default="20170101")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--benchmark", default="000905.SH")
    parser.add_argument("--reference-dir", default="data/reference")
    parser.add_argument("--bar-cache-dir", default="data/qmt_bars")
    parser.add_argument("--output", default="output/v5_composite_oos")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--min-price", type=float, default=3.0)
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)
    parser.add_argument("--min-listing-sessions", type=int, default=120)
    parser.add_argument("--min-symbol-coverage", type=float, default=0.98)
    return parser.parse_args()


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


def _eligible_mask(
    *,
    raw_close: pd.DataFrame,
    amount: pd.DataFrame,
    dates: pd.DatetimeIndex,
    reference: ReferenceData,
    universe: list[str],
    min_price: float,
    min_amount: float,
    min_listing_sessions: int,
    amount_window: int,
) -> pd.DataFrame:
    avg_amount = amount.rolling(amount_window, min_periods=amount_window).mean().reindex(dates)
    mask = raw_close.reindex(dates).ge(float(min_price)) & avg_amount.ge(float(min_amount))
    columns = mask.columns
    for ts in dates:
        if ts not in reference.st_dates:
            raise RuntimeError(f"missing ST snapshot on V5 score date {ts.date()}")
        members = set(
            reference.filter_members(
                universe,
                ts,
                min_listing_sessions=min_listing_sessions,
            )
        )
        allowed = members.difference(reference.st_codes(ts))
        mask.loc[ts, :] &= columns.isin(allowed)
    return mask


def main() -> int:
    args = parse_args()
    if args.train_years <= 0 or args.top_n <= 0 or args.rebalance_days <= 0:
        raise ValueError("train-years, top-n and rebalance-days must be positive")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(args.factor_observations)
    required_observation_columns = {
        "factor",
        "horizon",
        "date",
        "rank_ic",
        "top_bottom_spread",
    }
    missing = sorted(required_observation_columns.difference(observations.columns))
    if missing:
        raise ValueError(f"factor observations missing columns: {', '.join(missing)}")
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    observations = observations.dropna(subset=["date"])
    horizons = sorted(pd.to_numeric(observations["horizon"], errors="coerce").dropna().astype(int).unique())
    if not horizons or min(horizons) <= 0:
        raise ValueError("factor observations contain no positive horizons")
    max_forward_horizon = int(max(horizons))

    reference = ReferenceData.from_dir(args.reference_dir)
    reference_audit = reference.audit()
    if reference_audit.st_dates < reference_audit.calendar_sessions:
        raise RuntimeError("ST snapshots are incomplete; refusing V5 OOS")
    if reference_audit.limit_dates < reference_audit.calendar_sessions:
        raise RuntimeError("price-limit snapshots are incomplete; refusing V5 OOS")

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

    calendar = pd.DatetimeIndex(bars[args.benchmark].index).normalize().sort_values().unique()
    calendar = calendar[(calendar >= pd.Timestamp(args.data_start)) & (calendar <= pd.Timestamp(args.end))]
    first_validation = pd.Timestamp(args.start).year + int(args.train_years)
    last_validation = pd.Timestamp(args.end).year
    folds = annual_folds(
        first_validation,
        last_validation,
        train_years=args.train_years,
    )
    purged = select_purged_folds(
        observations,
        calendar,
        folds,
        max_forward_horizon=max_forward_horizon,
    )
    selected_names = selected_factor_union(purged)
    if not selected_names:
        raise RuntimeError("no factor survived training-only V5 selection")

    strategy = StrategyConfig(
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        min_price=args.min_price,
        min_amount=args.min_amount,
        min_listing_sessions=args.min_listing_sessions,
        risk_off_exposure=0.0,
    )
    fold_context: dict[int, pd.Timestamp] = {}
    for row in purged:
        context_start, _, truncated = context_start_for_window(
            bars,
            args.benchmark,
            strategy,
            row.fold.validation_start,
        )
        if truncated:
            raise RuntimeError(f"fold {row.fold.validation_year} has truncated warmup")
        fold_context[row.fold.validation_year] = context_start
    needed_start = min(fold_context.values())
    needed_end = pd.Timestamp(args.end)
    score_dates = calendar[(calendar >= needed_start) & (calendar <= needed_end)]

    stock_bars = {code: frame for code, frame in bars.items() if code != args.benchmark}
    close = _panel(stock_bars, "close", calendar)
    amount = _panel(stock_bars, "amount", calendar)
    raw_close = _panel(raw_bars, "close", calendar)
    benchmark_close = bars[args.benchmark]["close"].reindex(calendar).ffill()
    factor_cfg = V5FactorConfig()
    eligible = _eligible_mask(
        raw_close=raw_close,
        amount=amount,
        dates=score_dates,
        reference=reference,
        universe=universe,
        min_price=args.min_price,
        min_amount=args.min_amount,
        min_listing_sessions=args.min_listing_sessions,
        amount_window=factor_cfg.amount_window,
    )

    ranked_panels: dict[str, pd.DataFrame] = {}
    wanted = set(selected_names)
    for factor_name, factor_panel in iter_v5_raw_factors(
        close,
        amount,
        benchmark_close,
        factor_cfg,
    ):
        if factor_name not in wanted:
            del factor_panel
            continue
        print(f"Preparing selected factor panel: {factor_name}")
        sampled = factor_panel.reindex(score_dates).where(eligible)
        ranked_panels[factor_name] = normalize_factor(
            sampled,
            lower=factor_cfg.winsor_lower,
            upper=factor_cfg.winsor_upper,
        )
        del sampled, factor_panel
        gc.collect()
    missing_selected = sorted(wanted.difference(ranked_panels))
    if missing_selected:
        raise RuntimeError(f"selected factor panels missing: {', '.join(missing_selected)}")

    fold_rows: list[dict] = []
    fold_selection_rows: list[dict] = []
    trade_parts: list[pd.DataFrame] = []
    stitched_parts: list[pd.Series] = []
    chained_capital = 1.0

    for row in purged:
        fold = row.fold
        val_end = min(fold.validation_end, pd.Timestamp(args.end))
        spec = row.selection.spec
        score = apply_composite(ranked_panels, spec)
        risk_on = pd.Series(True, index=score.index, dtype=bool)
        result = run_window_backtest(
            bars,
            args.benchmark,
            strategy,
            CostConfig(),
            trade_start=fold.validation_start,
            trade_end=val_end,
            reference=reference,
            strict_reference=True,
            limit_reference_bars=raw_bars,
            score_override=score,
            risk_on_override=risk_on,
        )
        metrics = result.metrics
        if not metrics:
            raise RuntimeError(f"fold {fold.validation_year} returned no metrics")
        equity = result.equity["equity"].dropna()
        if equity.empty:
            raise RuntimeError(f"fold {fold.validation_year} returned no equity")
        normalized = equity / float(equity.iloc[0]) * chained_capital
        if stitched_parts and not normalized.empty:
            normalized = normalized.iloc[1:]
        stitched_parts.append(normalized)
        chained_capital *= float(metrics.get("multiple", 1.0))

        selection_payload = row.to_dict()
        fold_selection_rows.append(selection_payload)
        fold_rows.append(
            {
                "validation_year": int(fold.validation_year),
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "evidence_end": str(row.evidence_end.date()),
                "validation_start": str(fold.validation_start.date()),
                "validation_end": str(val_end.date()),
                "selected_factors": ",".join(row.selection.selected_factors),
                "weights": json.dumps(spec.weights, ensure_ascii=False, sort_keys=True),
                "validation_return": metrics.get("total_return"),
                "validation_multiple": metrics.get("multiple"),
                "validation_cagr": metrics.get("cagr"),
                "validation_max_drawdown": metrics.get("max_drawdown"),
                "validation_sharpe": metrics.get("sharpe"),
                "validation_calmar": metrics.get("calmar"),
                "trade_count": metrics.get("trade_count"),
                "blocked_st_candidates": metrics.get("blocked_st_candidates"),
                "blocked_limit_buys": metrics.get("blocked_limit_buys"),
                "blocked_limit_sells": metrics.get("blocked_limit_sells"),
                "blocked_suspended": metrics.get("blocked_suspended"),
                "missing_limit_rows": metrics.get("missing_limit_rows"),
                "missing_st_dates": metrics.get("missing_st_dates"),
                "missing_limit_dates": metrics.get("missing_limit_dates"),
            }
        )
        if not result.trades.empty:
            trades = result.trades.copy()
            trades.insert(0, "validation_year", int(fold.validation_year))
            trade_parts.append(trades)
        del score, risk_on, result
        gc.collect()

    folds_df = pd.DataFrame(fold_rows)
    assert_no_future_training(folds_df)
    folds_df.to_csv(out / "walk_forward_folds.csv", index=False, encoding="utf-8-sig")
    with (out / "fold_selections.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_selection_rows, handle, ensure_ascii=False, indent=2)
    adjusted_report.to_csv(out / "adjusted_coverage.csv", index=False, encoding="utf-8-sig")
    raw_report.to_csv(out / "raw_coverage.csv", index=False, encoding="utf-8-sig")
    if trade_parts:
        pd.concat(trade_parts, ignore_index=True).to_csv(
            out / "oos_trades.csv", index=False, encoding="utf-8-sig"
        )

    stitched = pd.concat(stitched_parts).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    stitched.to_frame("oos_equity").to_csv(out / "oos_equity.csv", encoding="utf-8-sig")
    oos_metrics = calculate_metrics(stitched)
    oos_metrics.update(
        {
            "symbol_coverage_ratio": adjusted_coverage,
            "raw_symbol_coverage_ratio": raw_coverage,
            "fold_count": int(len(folds_df)),
            "positive_oos_folds": int((folds_df["validation_return"] > 0).sum()),
            "method": "purged_training_only_v5_composite_reset_folds",
            "max_forward_horizon_purged": max_forward_horizon,
            "stock_selection_only": True,
            "timing_override": "always_on",
            "top_n": int(args.top_n),
            "rebalance_days": int(args.rebalance_days),
        }
    )
    with (out / "oos_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(oos_metrics, handle, ensure_ascii=False, indent=2)

    gates = evaluate_v5_gates(oos_metrics, folds_df)
    alpha_gate_names = {
        "positive_oos_return",
        "oos_sharpe",
        "oos_max_drawdown",
        "non_disastrous_oos_folds",
    }
    gates["alpha_discovery_passed"] = all(
        gate["passed"] for gate in gates["gates"] if gate["name"] in alpha_gate_names
    )
    gates["promotion_pending_stress_and_robustness"] = not gates["passed"]
    with (out / "v5_gate_report.json").open("w", encoding="utf-8") as handle:
        json.dump(gates, handle, ensure_ascii=False, indent=2)

    manifest = {
        "passed": True,
        "data_start": args.data_start,
        "research_start": args.start,
        "end": args.end,
        "benchmark": args.benchmark,
        "strict_reference": True,
        "strict_point_in_time": True,
        "strict_st": True,
        "strict_price_limits": True,
        "strict_suspension_execution": True,
        "stock_selection_only": True,
        "timing_override": "always_on",
        "factor_observations": str(args.factor_observations),
        "factor_horizons": horizons,
        "max_forward_horizon_purged": max_forward_horizon,
        "selected_factor_union": list(selected_names),
        "fold_count": int(len(folds_df)),
        "adjusted_symbol_coverage": adjusted_coverage,
        "raw_symbol_coverage": raw_coverage,
        "min_symbol_coverage": float(args.min_symbol_coverage),
        "min_price": float(args.min_price),
        "min_amount": float(args.min_amount),
        "min_listing_sessions": int(args.min_listing_sessions),
        "strategy": {
            "top_n": int(args.top_n),
            "rebalance_days": int(args.rebalance_days),
            "execution_delay_sessions": int(strategy.execution_delay_sessions),
        },
        "reference_audit": {
            "basic_symbols": reference_audit.basic_symbols,
            "st_dates": reference_audit.st_dates,
            "limit_dates": reference_audit.limit_dates,
            "calendar_sessions": reference_audit.calendar_sessions,
        },
    }
    with (out / "research_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(json.dumps(oos_metrics, ensure_ascii=False, indent=2))
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    print(folds_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
