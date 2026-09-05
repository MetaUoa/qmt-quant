from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from qmt_quant.ablation import leave_one_out_specs, pair_specs, single_factor_specs, summarize_ablation_metrics
from qmt_quant.backtest import _panel, calculate_metrics
from qmt_quant.composites import CompositeSpec, apply_composite
from qmt_quant.config import CostConfig, StrategyConfig
from qmt_quant.factor_attribution import leave_one_out_attribution, pair_interactions, summarize_attribution
from qmt_quant.factor_diagnostics import factor_observations, forward_return_panel, summarize_factor_observations
from qmt_quant.factors import V5FactorConfig, iter_v5_raw_factors, normalize_factor
from qmt_quant.neutralization import neutralize_panel
from qmt_quant.qmt_data import coverage_report, load_daily_bars, load_limit_reference_bars
from qmt_quant.quantile_profiles import summarize_tail_profiles, tail_linearity_score, tail_profile_observations
from qmt_quant.reference_data import ReferenceData
from qmt_quant.regime_weighting import (
    apply_regime_composite,
    classify_regimes,
    fit_regime_factor_weights,
    fit_regime_model,
)
from qmt_quant.windowed import context_start_for_window, run_window_backtest


B2_FACTORS = {
    "low_volatility",
    "low_downside_risk",
    "liquidity_stability",
    "short_reversal",
    "momentum_20_5",
    "momentum_60_5",
    "momentum_120_5",
    "trend_quality",
    "relative_strength_60_5",
    "residual_relative_strength_60_5",
}
B3_PROXY_FACTORS = {
    "low_volatility",
    "liquidity_stability",
    "momentum_20_5",
    "momentum_120_5",
    "trend_quality",
    "residual_relative_strength_60_5",
}
B4_COMPARE_FACTORS = {
    "momentum_60_5",
    "relative_strength_60_5",
    "residual_relative_strength_60_5",
}
B1_FOCUS_YEARS = {2022, 2024}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict full-market V5 B1-B6 research")
    parser.add_argument("--factor-observations", required=True)
    parser.add_argument("--fold-selections", required=True)
    parser.add_argument("--baseline-folds", required=True)
    parser.add_argument("--data-start", default="20170101")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20251231")
    parser.add_argument("--benchmark", default="000905.SH")
    parser.add_argument("--reference-dir", default="data/reference")
    parser.add_argument("--bar-cache-dir", default="data/qmt_bars")
    parser.add_argument("--output", default="output/v5_b_research")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--min-price", type=float, default=3.0)
    parser.add_argument("--min-amount", type=float, default=20_000_000.0)
    parser.add_argument("--min-listing-sessions", type=int, default=120)
    parser.add_argument("--min-symbol-coverage", type=float, default=0.98)
    parser.add_argument("--min-symbols-per-date", type=int, default=50)
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
            raise RuntimeError(f"missing ST snapshot on B research date {ts.date()}")
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


def _assert_strict_metrics(metrics: Mapping[str, object], label: str) -> None:
    for key in ("missing_limit_rows", "missing_st_dates", "missing_limit_dates"):
        value = int(metrics.get(key, 0) or 0)
        if value != 0:
            raise RuntimeError(f"{label} has {key}={value}; refusing research result")


def _reference_availability(stock_basic: pd.DataFrame) -> dict[str, object]:
    columns = {str(column) for column in stock_basic.columns}
    industry_fields = sorted(columns.intersection({"industry", "industryClassification", "industry_classification"}))
    market_cap_fields = sorted(
        columns.intersection(
            {
                "total_mv",
                "circ_mv",
                "market_cap",
                "float_market_cap",
                "total_market_cap",
            }
        )
    )
    return {
        "stock_basic_columns": sorted(columns),
        "industry_neutralization_available": bool(industry_fields),
        "industry_fields": industry_fields,
        "market_cap_neutralization_available": bool(market_cap_fields),
        "market_cap_fields": market_cap_fields,
        "liquidity_proxy_neutralization_available": True,
        "liquidity_proxy": "log_20_session_average_amount",
        "note": (
            "Frozen BaoStock artifacts do not contain a PIT market-cap series or historical "
            "industry snapshots unless the corresponding fields are explicitly present. "
            "Liquidity proxy results must not be described as market-cap neutralization."
        ),
    }


def _mean_row_correlation(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float | int]:
    dates = left.index.intersection(right.index)
    values: list[float] = []
    for ts in dates:
        pair = pd.concat([left.loc[ts].rename("left"), right.loc[ts].rename("right")], axis=1).dropna()
        if len(pair) < 20:
            continue
        value = pair["left"].corr(pair["right"])
        if pd.notna(value):
            values.append(float(value))
    return {
        "dates": int(len(values)),
        "mean_cross_sectional_rank_correlation": float(np.mean(values)) if values else float("nan"),
        "median_cross_sectional_rank_correlation": float(np.median(values)) if values else float("nan"),
    }


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _run_score(
    *,
    bars: dict[str, pd.DataFrame],
    raw_bars: dict[str, pd.DataFrame],
    benchmark: str,
    strategy: StrategyConfig,
    reference: ReferenceData,
    score: pd.DataFrame,
    trade_start,
    trade_end,
    label: str,
):
    risk_on = pd.Series(True, index=score.index, dtype=bool)
    result = run_window_backtest(
        bars,
        benchmark,
        strategy,
        CostConfig(),
        trade_start=trade_start,
        trade_end=trade_end,
        reference=reference,
        strict_reference=True,
        limit_reference_bars=raw_bars,
        score_override=score,
        risk_on_override=risk_on,
    )
    if not result.metrics:
        raise RuntimeError(f"{label} returned no metrics")
    _assert_strict_metrics(result.metrics, label)
    return result


def _stitch_fold_equity(parts: list[pd.Series]) -> pd.Series:
    stitched_parts: list[pd.Series] = []
    chained = 1.0
    for equity in parts:
        clean = equity.dropna()
        if clean.empty:
            continue
        normalized = clean / float(clean.iloc[0]) * chained
        if stitched_parts:
            normalized = normalized.iloc[1:]
        if normalized.empty:
            continue
        stitched_parts.append(normalized)
        chained = float(normalized.iloc[-1])
    if not stitched_parts:
        return pd.Series(dtype=float)
    stitched = pd.concat(stitched_parts).sort_index()
    return stitched[~stitched.index.duplicated(keep="last")]


def _fold_spec(selection: Mapping[str, object]) -> CompositeSpec:
    weights = {str(name): float(value) for name, value in dict(selection["weights"]).items()}
    if not weights:
        raise RuntimeError("frozen fold selection has no weights")
    return CompositeSpec(name="frozen_v5_a", weights=weights)


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(args.factor_observations)
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    observations = observations.dropna(subset=["date"])
    with Path(args.fold_selections).open("r", encoding="utf-8-sig") as handle:
        selections = json.load(handle)
    if not isinstance(selections, list) or len(selections) != 5:
        raise RuntimeError("expected exactly five frozen 2021-2025 fold selections")
    baseline_folds = pd.read_csv(args.baseline_folds)
    baseline_folds["validation_year"] = pd.to_numeric(
        baseline_folds["validation_year"], errors="raise"
    ).astype(int)
    baseline_by_year = baseline_folds.set_index("validation_year")

    reference = ReferenceData.from_dir(args.reference_dir)
    reference_audit = reference.audit()
    if reference_audit.st_dates < reference_audit.calendar_sessions:
        raise RuntimeError("ST snapshots are incomplete; refusing B research")
    if reference_audit.limit_dates < reference_audit.calendar_sessions:
        raise RuntimeError("price-limit snapshots are incomplete; refusing B research")

    universe = reference.codes_ever_active(args.start, args.end)
    if not universe:
        raise RuntimeError("point-in-time historical universe is empty")
    all_codes = list(dict.fromkeys(universe + [args.benchmark]))
    range_cache = Path(args.bar_cache_dir) / f"front_{args.data_start}_{args.end}"
    bars = load_daily_bars(all_codes, args.data_start, args.end, cache_dir=range_cache)
    adjusted_coverage, adjusted_report = _coverage_or_fail(
        "adjusted", universe, bars, args.min_symbol_coverage
    )
    if args.benchmark not in bars or bars[args.benchmark].empty:
        raise RuntimeError(f"benchmark {args.benchmark} is missing")
    raw_cache = Path(args.bar_cache_dir) / f"none_limits_{args.data_start}_{args.end}"
    raw_bars = load_limit_reference_bars(
        universe, args.data_start, args.end, cache_dir=raw_cache
    )
    raw_coverage, raw_report = _coverage_or_fail(
        "raw", universe, raw_bars, args.min_symbol_coverage
    )

    calendar = pd.DatetimeIndex(bars[args.benchmark].index).normalize().sort_values().unique()
    calendar = calendar[(calendar >= pd.Timestamp(args.data_start)) & (calendar <= pd.Timestamp(args.end))]
    strategy = StrategyConfig(
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        min_price=args.min_price,
        min_amount=args.min_amount,
        min_listing_sessions=args.min_listing_sessions,
        risk_off_exposure=0.0,
    )
    fold_context: dict[int, pd.Timestamp] = {}
    for selection in selections:
        year = int(selection["validation_year"])
        context_start, _, truncated = context_start_for_window(
            bars, args.benchmark, strategy, selection["validation_start"]
        )
        if truncated:
            raise RuntimeError(f"fold {year} has truncated warmup")
        fold_context[year] = context_start

    stock_bars = {code: frame for code, frame in bars.items() if code != args.benchmark}
    close = _panel(stock_bars, "close", calendar)
    amount = _panel(stock_bars, "amount", calendar)
    raw_close = _panel(raw_bars, "close", calendar)
    benchmark_close = bars[args.benchmark]["close"].reindex(calendar).ffill()
    factor_cfg = V5FactorConfig()

    research_start = pd.Timestamp(args.start)
    research_start_pos = max(
        int(calendar.searchsorted(research_start, side="left")),
        int(factor_cfg.warmup),
    )
    research_end_pos = int(calendar.searchsorted(pd.Timestamp(args.end), side="right"))
    sample_dates = calendar[research_start_pos:research_end_pos: args.rebalance_days]
    if len(sample_dates) == 0:
        raise RuntimeError("no B research sample dates remain after warmup")
    analysis_start = min(pd.Timestamp(sample_dates[0]), min(fold_context.values()))
    analysis_dates = calendar[(calendar >= analysis_start) & (calendar <= pd.Timestamp(args.end))]
    eligible = _eligible_mask(
        raw_close=raw_close,
        amount=amount,
        dates=analysis_dates,
        reference=reference,
        universe=universe,
        min_price=args.min_price,
        min_amount=args.min_amount,
        min_listing_sessions=args.min_listing_sessions,
        amount_window=factor_cfg.amount_window,
    )
    sample_mask = eligible.reindex(sample_dates)
    avg_amount = amount.rolling(
        factor_cfg.amount_window, min_periods=factor_cfg.amount_window
    ).mean()
    liquidity_proxy = np.log1p(avg_amount.clip(lower=0.0)).reindex(sample_dates).where(sample_mask)
    forward_20 = forward_return_panel(close, 20).reindex(sample_dates).where(sample_mask)

    wanted_execution: set[str] = set()
    for selection in selections:
        wanted_execution.update(str(name) for name in dict(selection["weights"]))
    ranked_execution: dict[str, pd.DataFrame] = {}
    diagnostic_ranked: dict[str, pd.DataFrame] = {}
    b2_rows: list[pd.DataFrame] = []
    b3_rows: list[dict] = []
    b4_summary_rows: list[dict] = []

    for factor_name, raw_factor in iter_v5_raw_factors(
        close, amount, benchmark_close, factor_cfg
    ):
        needs_execution = factor_name in wanted_execution
        needs_diagnostic = factor_name in B2_FACTORS or factor_name in B3_PROXY_FACTORS
        if not needs_execution and not needs_diagnostic:
            del raw_factor
            continue
        print(f"B research factor: {factor_name}")
        analysis_factor = raw_factor.reindex(analysis_dates).where(eligible)
        if needs_execution:
            ranked_execution[factor_name] = normalize_factor(
                analysis_factor,
                lower=factor_cfg.winsor_lower,
                upper=factor_cfg.winsor_upper,
            )

        sampled = raw_factor.reindex(sample_dates).where(sample_mask)
        if factor_name in B2_FACTORS:
            tails = tail_profile_observations(
                sampled,
                forward_20,
                fractions=(0.05, 0.10, 0.20),
                min_symbols=args.min_symbols_per_date,
            )
            tail_summary = summarize_tail_profiles(tails, fractions=(0.05, 0.10, 0.20))
            tail_summary.insert(0, "factor", factor_name)
            tail_summary["tail_linearity_score"] = tail_linearity_score(tail_summary)
            b2_rows.append(tail_summary)

        if factor_name in B3_PROXY_FACTORS:
            before_obs = factor_observations(
                sampled,
                forward_20,
                min_symbols=args.min_symbols_per_date,
            )
            neutral = neutralize_panel(
                sampled,
                exposure_panels={"log_avg_amount": liquidity_proxy},
                min_symbols=args.min_symbols_per_date,
                min_coverage=0.95,
            )
            after_obs = factor_observations(
                neutral,
                forward_20,
                min_symbols=args.min_symbols_per_date,
            )
            before_summary = summarize_factor_observations(before_obs)
            after_summary = summarize_factor_observations(after_obs)
            b3_rows.append(
                {
                    "factor": factor_name,
                    "exposure": "log_20_session_average_amount",
                    "before_mean_rank_ic": before_summary["mean_rank_ic"],
                    "after_mean_rank_ic": after_summary["mean_rank_ic"],
                    "rank_ic_delta": after_summary["mean_rank_ic"] - before_summary["mean_rank_ic"],
                    "before_ic_ir": before_summary["ic_ir"],
                    "after_ic_ir": after_summary["ic_ir"],
                    "before_mean_top_bottom_spread": before_summary["mean_top_bottom_spread"],
                    "after_mean_top_bottom_spread": after_summary["mean_top_bottom_spread"],
                }
            )
            del neutral, before_obs, after_obs

        if factor_name in B4_COMPARE_FACTORS:
            obs = factor_observations(
                sampled,
                forward_20,
                min_symbols=args.min_symbols_per_date,
            )
            summary = summarize_factor_observations(obs)
            b4_summary_rows.append({"factor": factor_name, "horizon": 20, **summary})
            diagnostic_ranked[factor_name] = normalize_factor(
                sampled,
                lower=factor_cfg.winsor_lower,
                upper=factor_cfg.winsor_upper,
            )
            del obs
        del sampled, analysis_factor, raw_factor
        gc.collect()

    missing_execution = sorted(wanted_execution.difference(ranked_execution))
    if missing_execution:
        raise RuntimeError(f"missing execution factors: {', '.join(missing_execution)}")

    b2_summary = pd.concat(b2_rows, ignore_index=True) if b2_rows else pd.DataFrame()
    b2_summary.to_csv(out / "b2_tail_summary.csv", index=False, encoding="utf-8-sig")
    b3_proxy = pd.DataFrame(b3_rows)
    b3_proxy.to_csv(out / "b3_liquidity_proxy_neutralization.csv", index=False, encoding="utf-8-sig")
    b3_availability = _reference_availability(reference.stock_basic)
    _write_json(out / "b3_reference_availability.json", b3_availability)
    b4_summary = pd.DataFrame(b4_summary_rows)
    b4_summary.to_csv(out / "b4_factor_summary.csv", index=False, encoding="utf-8-sig")
    b4_correlations = {
        "legacy_vs_momentum": _mean_row_correlation(
            diagnostic_ranked["relative_strength_60_5"],
            diagnostic_ranked["momentum_60_5"],
        ),
        "residual_vs_momentum": _mean_row_correlation(
            diagnostic_ranked["residual_relative_strength_60_5"],
            diagnostic_ranked["momentum_60_5"],
        ),
        "residual_vs_legacy": _mean_row_correlation(
            diagnostic_ranked["residual_relative_strength_60_5"],
            diagnostic_ranked["relative_strength_60_5"],
        ),
    }
    _write_json(out / "b4_rank_correlations.json", b4_correlations)

    b1_attribution_parts: list[pd.DataFrame] = []
    b1_interaction_parts: list[pd.DataFrame] = []
    b1_summaries: list[dict] = []
    b6_parts: list[pd.DataFrame] = []
    b5_rows: list[dict] = []
    b5_weight_payload: list[dict] = []
    b5_equity_parts: list[pd.Series] = []

    for selection in selections:
        year = int(selection["validation_year"])
        val_start = pd.Timestamp(selection["validation_start"])
        val_end = min(pd.Timestamp(selection["validation_end"]), pd.Timestamp(args.end))
        spec = _fold_spec(selection)
        static_score = apply_composite(ranked_execution, spec)
        static = _run_score(
            bars=bars,
            raw_bars=raw_bars,
            benchmark=args.benchmark,
            strategy=strategy,
            reference=reference,
            score=static_score,
            trade_start=val_start,
            trade_end=val_end,
            label=f"B static fold {year}",
        )
        if year not in baseline_by_year.index:
            raise RuntimeError(f"baseline OOS artifact missing fold {year}")
        expected_return = float(baseline_by_year.loc[year, "validation_return"])
        actual_return = float(static.metrics["total_return"])
        if not np.isclose(actual_return, expected_return, atol=1e-10, rtol=1e-10):
            raise RuntimeError(
                f"frozen fold {year} return drifted: expected {expected_return}, got {actual_return}"
            )

        ablated_metrics: dict[str, Mapping[str, float]] = {}
        for removed, ablated_spec in leave_one_out_specs(spec).items():
            ablated_score = apply_composite(ranked_execution, ablated_spec)
            result = _run_score(
                bars=bars,
                raw_bars=raw_bars,
                benchmark=args.benchmark,
                strategy=strategy,
                reference=reference,
                score=ablated_score,
                trade_start=val_start,
                trade_end=val_end,
                label=f"B6 fold {year} without {removed}",
            )
            ablated_metrics[removed] = result.metrics
            del ablated_score, result
        b6 = summarize_ablation_metrics(static.metrics, ablated_metrics)
        b6.insert(0, "validation_year", year)
        b6_parts.append(b6)

        if year in B1_FOCUS_YEARS:
            attribution = leave_one_out_attribution(
                actual_return,
                {name: float(metrics["total_return"]) for name, metrics in ablated_metrics.items()},
            )
            attribution.insert(0, "validation_year", year)
            b1_attribution_parts.append(attribution)

            singles: dict[str, float] = {}
            for factor, single_spec in single_factor_specs(spec).items():
                score = apply_composite(ranked_execution, single_spec)
                result = _run_score(
                    bars=bars,
                    raw_bars=raw_bars,
                    benchmark=args.benchmark,
                    strategy=strategy,
                    reference=reference,
                    score=score,
                    trade_start=val_start,
                    trade_end=val_end,
                    label=f"B1 fold {year} single {factor}",
                )
                singles[factor] = float(result.metrics["total_return"])
                del score, result
            pairs: dict[tuple[str, str], float] = {}
            for pair, pair_spec in pair_specs(spec).items():
                score = apply_composite(ranked_execution, pair_spec)
                result = _run_score(
                    bars=bars,
                    raw_bars=raw_bars,
                    benchmark=args.benchmark,
                    strategy=strategy,
                    reference=reference,
                    score=score,
                    trade_start=val_start,
                    trade_end=val_end,
                    label=f"B1 fold {year} pair {pair[0]}+{pair[1]}",
                )
                pairs[pair] = float(result.metrics["total_return"])
                del score, result
            interactions = pair_interactions(singles, pairs)
            interactions.insert(0, "validation_year", year)
            b1_interaction_parts.append(interactions)
            summary = summarize_attribution(attribution, interactions)
            summary["validation_year"] = year
            b1_summaries.append(summary)

        train_start = pd.Timestamp(selection["train_start"])
        evidence_end = pd.Timestamp(selection["evidence_end"])
        regime_model = fit_regime_model(
            benchmark_close,
            train_start=train_start,
            train_end=evidence_end,
            min_dates=120,
        )
        regimes = classify_regimes(benchmark_close, regime_model)
        fitted_weights = fit_regime_factor_weights(
            observations,
            regimes,
            train_start=train_start,
            train_end=evidence_end,
            factors=list(spec.weights),
            horizon=20,
            min_regime_dates=12,
        )
        regime_score = apply_regime_composite(
            {name: ranked_execution[name] for name in spec.weights},
            regimes,
            fitted_weights,
        )
        regime_result = _run_score(
            bars=bars,
            raw_bars=raw_bars,
            benchmark=args.benchmark,
            strategy=strategy,
            reference=reference,
            score=regime_score,
            trade_start=val_start,
            trade_end=val_end,
            label=f"B5 regime fold {year}",
        )
        b5_equity_parts.append(regime_result.equity["equity"].copy())
        b5_rows.append(
            {
                "validation_year": year,
                "static_return": actual_return,
                "regime_return": regime_result.metrics.get("total_return"),
                "return_delta": float(regime_result.metrics.get("total_return", np.nan)) - actual_return,
                "static_sharpe": static.metrics.get("sharpe"),
                "regime_sharpe": regime_result.metrics.get("sharpe"),
                "static_max_drawdown": static.metrics.get("max_drawdown"),
                "regime_max_drawdown": regime_result.metrics.get("max_drawdown"),
                "trade_count": regime_result.metrics.get("trade_count"),
            }
        )
        b5_weight_payload.append(
            {
                "validation_year": year,
                "train_start": str(train_start.date()),
                "evidence_end": str(evidence_end.date()),
                "vol_threshold": regime_model.vol_threshold,
                "global_weights": fitted_weights.global_weights,
                "weights_by_regime": fitted_weights.weights_by_regime,
                "dates_by_regime": fitted_weights.dates_by_regime,
            }
        )
        del static_score, static, regime_score, regime_result, regimes
        gc.collect()

    b1_attribution = (
        pd.concat(b1_attribution_parts, ignore_index=True) if b1_attribution_parts else pd.DataFrame()
    )
    b1_interactions = (
        pd.concat(b1_interaction_parts, ignore_index=True) if b1_interaction_parts else pd.DataFrame()
    )
    b1_attribution.to_csv(out / "b1_focus_year_attribution.csv", index=False, encoding="utf-8-sig")
    b1_interactions.to_csv(out / "b1_focus_year_interactions.csv", index=False, encoding="utf-8-sig")
    _write_json(out / "b1_focus_year_summary.json", b1_summaries)

    b6_all = pd.concat(b6_parts, ignore_index=True) if b6_parts else pd.DataFrame()
    b6_all.to_csv(out / "b6_ablation.csv", index=False, encoding="utf-8-sig")
    b6_aggregate = pd.DataFrame()
    if not b6_all.empty:
        b6_aggregate = (
            b6_all.groupby("factor_removed", as_index=False)
            .agg(
                folds=("validation_year", "count"),
                mean_return_contribution=("return_contribution", "mean"),
                median_return_contribution=("return_contribution", "median"),
                mean_sharpe_contribution=("sharpe_contribution", "mean"),
                mean_drawdown_cost=("drawdown_cost", "mean"),
            )
            .sort_values("mean_return_contribution")
            .reset_index(drop=True)
        )
    b6_aggregate.to_csv(out / "b6_ablation_aggregate.csv", index=False, encoding="utf-8-sig")

    b5_folds = pd.DataFrame(b5_rows)
    b5_folds.to_csv(out / "b5_regime_fold_metrics.csv", index=False, encoding="utf-8-sig")
    _write_json(out / "b5_regime_weights.json", b5_weight_payload)
    b5_equity = _stitch_fold_equity(b5_equity_parts)
    b5_metrics = calculate_metrics(b5_equity) if not b5_equity.empty else {}
    b5_metrics.update(
        {
            "fold_count": int(len(b5_folds)),
            "positive_folds": int((b5_folds["regime_return"] > 0.0).sum()) if not b5_folds.empty else 0,
            "method": "training_only_regime_factor_weighting",
            "timing_override": "always_on",
            "stock_selection_layer_only": True,
        }
    )
    _write_json(out / "b5_regime_oos_metrics.json", b5_metrics)
    b5_equity.to_frame("oos_equity").to_csv(out / "b5_regime_oos_equity.csv", encoding="utf-8-sig")

    adjusted_report.to_csv(out / "adjusted_coverage.csv", index=False, encoding="utf-8-sig")
    raw_report.to_csv(out / "raw_coverage.csv", index=False, encoding="utf-8-sig")

    b4_residual = b4_summary.loc[
        b4_summary["factor"].eq("residual_relative_strength_60_5")
    ]
    residual_mean_ic = (
        float(b4_residual.iloc[0]["mean_rank_ic"]) if not b4_residual.empty else float("nan")
    )
    largest_drag = None
    if not b6_aggregate.empty:
        largest_drag = str(b6_aggregate.iloc[0]["factor_removed"])
    decision = {
        "v5_a_static_oos_return": float(
            np.prod(1.0 + pd.to_numeric(baseline_folds["validation_return"], errors="coerce")) - 1.0
        ),
        "b5_regime_oos_return": b5_metrics.get("total_return"),
        "b5_regime_sharpe": b5_metrics.get("sharpe"),
        "b5_regime_max_drawdown": b5_metrics.get("max_drawdown"),
        "b6_largest_mean_drag_factor": largest_drag,
        "b4_residual_relative_strength_mean_rank_ic_20": residual_mean_ic,
        "b3_industry_neutralization_available": b3_availability["industry_neutralization_available"],
        "b3_market_cap_neutralization_available": b3_availability["market_cap_neutralization_available"],
        "b3_liquidity_proxy_completed": True,
        "stock_selection_alpha_established": bool(
            np.isfinite(float(b5_metrics.get("total_return", np.nan)))
            and float(b5_metrics.get("total_return", np.nan)) > 0.0
            and np.isfinite(float(b5_metrics.get("sharpe", np.nan)))
            and float(b5_metrics.get("sharpe", np.nan)) > 0.0
        ),
    }
    _write_json(out / "research_decision.json", decision)

    manifest = {
        "passed": True,
        "data_start": args.data_start,
        "research_start": args.start,
        "end": args.end,
        "benchmark": args.benchmark,
        "universe_symbols": int(len(universe)),
        "adjusted_symbol_coverage": adjusted_coverage,
        "raw_symbol_coverage": raw_coverage,
        "strict_reference": True,
        "strict_point_in_time": True,
        "strict_st": True,
        "strict_price_limits": True,
        "strict_suspension_execution": True,
        "timing_override": "always_on",
        "research_dates": int(len(sample_dates)),
        "b1_focus_years": sorted(B1_FOCUS_YEARS),
        "b2_tail_fractions": [0.05, 0.10, 0.20],
        "b3": b3_availability,
        "b4_residual_relative_strength": True,
        "b5_training_only_regime_weights": True,
        "b6_frozen_weight_ablation": True,
        "reference_audit": {
            "basic_symbols": int(reference_audit.basic_symbols),
            "st_dates": int(reference_audit.st_dates),
            "limit_dates": int(reference_audit.limit_dates),
            "calendar_sessions": int(reference_audit.calendar_sessions),
        },
    }
    _write_json(out / "research_manifest.json", manifest)

    print(json.dumps(decision, ensure_ascii=False, indent=2, default=_json_default))
    print("\nB1 focus-year attribution")
    print(b1_attribution.to_string(index=False) if not b1_attribution.empty else "none")
    print("\nB2 tail summary")
    print(b2_summary.to_string(index=False) if not b2_summary.empty else "none")
    print("\nB3 liquidity-proxy neutralization")
    print(b3_proxy.to_string(index=False) if not b3_proxy.empty else "none")
    print("\nB4 factor summary")
    print(b4_summary.to_string(index=False) if not b4_summary.empty else "none")
    print("\nB5 regime OOS")
    print(json.dumps(b5_metrics, ensure_ascii=False, indent=2, default=_json_default))
    print("\nB6 aggregate ablation")
    print(b6_aggregate.to_string(index=False) if not b6_aggregate.empty else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
