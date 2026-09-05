from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from qmt_quant.backtest import _panel, calculate_metrics
from qmt_quant.composites import apply_composite
from qmt_quant.config import CostConfig, StrategyConfig
from qmt_quant.core_alpha import CORE_ALPHA_FACTORS, CoreAlphaPolicy, select_core_alpha
from qmt_quant.factor_diagnostics import factor_observations, forward_return_panel
from qmt_quant.factors import V5FactorConfig, iter_v5_raw_factors, normalize_factor
from qmt_quant.holdout import FrozenCandidate, freeze_candidate_manifest
from qmt_quant.nested_walk_forward import choose_inner_candidate, nested_annual_folds, purge_nested_fold
from qmt_quant.neutralized_alpha import NeutralizationInputs, neutralize_factor_panels
from qmt_quant.pit_exposures import asof_industry_panel
from qmt_quant.qmt_data import coverage_report, load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData
from qmt_quant.windowed import context_start_for_window, run_window_backtest


VARIANTS = ("raw", "liquidity", "industry", "industry_size_liquidity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict nested full-market V5 C research")
    p.add_argument("--exposure-root", required=True)
    p.add_argument("--industry-snapshots", required=True)
    p.add_argument("--data-start", default="20170101")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20251231")
    p.add_argument("--benchmark", default="000905.SH")
    p.add_argument("--reference-dir", default="data/reference")
    p.add_argument("--bar-cache-dir", default="data/qmt_bars")
    p.add_argument("--output", default="output/v5_c_nested")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--rebalance-days", type=int, default=5)
    p.add_argument("--min-price", type=float, default=3.0)
    p.add_argument("--min-amount", type=float, default=20_000_000.0)
    p.add_argument("--min-listing-sessions", type=int, default=120)
    p.add_argument("--min-symbol-coverage", type=float, default=0.98)
    p.add_argument("--min-exposure-coverage", type=float, default=0.95)
    p.add_argument("--min-symbols-per-date", type=int, default=50)
    return p.parse_args()


def _write_json(path: Path, payload) -> None:
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return str(pd.Timestamp(value))
        raise TypeError(type(value).__name__)

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def _coverage_or_fail(label: str, universe: list[str], bars: dict[str, pd.DataFrame], threshold: float) -> float:
    report = coverage_report(universe, bars)
    ratio = float(report["loaded"].mean()) if not report.empty else 0.0
    if ratio < float(threshold):
        raise RuntimeError(f"{label} symbol coverage {ratio:.4%} is below required {threshold:.4%}")
    return ratio


def _eligible_mask(
    *,
    raw_close: pd.DataFrame,
    amount: pd.DataFrame,
    suspend: pd.DataFrame,
    dates: pd.DatetimeIndex,
    reference: ReferenceData,
    universe: list[str],
    min_price: float,
    min_amount: float,
    min_listing_sessions: int,
    amount_window: int,
) -> pd.DataFrame:
    avg_amount = amount.rolling(amount_window, min_periods=amount_window).mean().reindex(dates)
    same_day_amount = amount.reindex(dates)
    same_day_suspend = suspend.reindex(dates).apply(pd.to_numeric, errors="coerce")
    tradable = same_day_suspend.eq(0.0) & same_day_amount.gt(0.0)
    mask = (
        raw_close.reindex(dates).ge(float(min_price))
        & avg_amount.ge(float(min_amount))
        & tradable
    )
    columns = mask.columns
    for ts in dates:
        if ts not in reference.st_dates:
            raise RuntimeError(f"missing ST snapshot on C research date {ts.date()}")
        members = set(reference.filter_members(universe, ts, min_listing_sessions=min_listing_sessions))
        allowed = members.difference(reference.st_codes(ts))
        mask.loc[ts, :] &= columns.isin(allowed)
    return mask


def _assert_strict_metrics(metrics: Mapping[str, object], label: str) -> None:
    for key in ("missing_limit_rows", "missing_st_dates", "missing_limit_dates"):
        value = int(metrics.get(key, 0) or 0)
        if value != 0:
            raise RuntimeError(f"{label} has {key}={value}; refusing research result")


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
    _assert_strict_metrics(result.metrics, label)
    return result


def _signal_dates_for_window(
    bars: dict[str, pd.DataFrame],
    benchmark: str,
    strategy: StrategyConfig,
    *,
    trade_start,
    trade_end,
) -> pd.DatetimeIndex:
    context_start, _, truncated = context_start_for_window(bars, benchmark, strategy, trade_start)
    if truncated:
        raise RuntimeError(f"window {trade_start} has truncated warmup")
    calendar = pd.DatetimeIndex(bars[benchmark].index).sort_values().unique()
    window = calendar[(calendar >= context_start) & (calendar <= pd.Timestamp(trade_end))]
    delay = max(int(strategy.execution_delay_sessions), 1)
    start_i = min(strategy.warmup + delay - 1, max(len(window) - 1, 0))
    signals = []
    for i in range(start_i, len(window)):
        if ((i - start_i) % strategy.rebalance_days) == 0 and i > 0:
            signals.append(window[i - delay])
    return pd.DatetimeIndex(signals)


def _load_size_panel(root: Path, dates: pd.DatetimeIndex, codes: list[str]) -> tuple[pd.DataFrame, dict]:
    manifests = sorted(root.rglob("pit_exposure_manifest.json"))
    if len(manifests) != 20:
        raise RuntimeError(f"expected exactly 20 PIT exposure manifests, found {len(manifests)}")
    manifest_rows = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    indexes = {int(row["shard_index"]) for row in manifest_rows}
    if indexes != set(range(20)) or any(not bool(row.get("strict_ready")) for row in manifest_rows):
        raise RuntimeError("PIT exposure shard set is incomplete or not strict_ready")

    wanted = set(codes)
    series: dict[str, pd.Series] = {}
    for path in root.rglob("*.parquet"):
        if path.name == "stock_basic_shard.parquet":
            continue
        code = path.stem
        if code not in wanted:
            continue
        frame = pd.read_parquet(path, columns=["date", "log_float_market_cap"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        values = pd.to_numeric(frame["log_float_market_cap"], errors="coerce")
        values.index = frame["date"]
        series[code] = values[~values.index.duplicated(keep="last")].sort_index().reindex(dates)
    panel = pd.DataFrame(series, index=dates).reindex(columns=codes)
    coverage = float(panel.notna().sum().sum() / max(panel.size, 1))
    return panel, {
        "manifest_count": len(manifests),
        "symbol_files": len(series),
        "cell_coverage": coverage,
        "source": "baostock_turnover_implied_float_cap",
    }


def _variant_observations(
    panels: Mapping[str, pd.DataFrame],
    forward_20: pd.DataFrame,
    sample_dates: pd.DatetimeIndex,
    *,
    min_symbols: int,
) -> pd.DataFrame:
    rows = []
    for factor in CORE_ALPHA_FACTORS:
        obs = factor_observations(
            panels[factor].reindex(sample_dates),
            forward_20.reindex(sample_dates),
            min_symbols=min_symbols,
        )
        obs.insert(0, "factor", factor)
        obs.insert(1, "horizon", 20)
        rows.append(obs)
    return pd.concat(rows, ignore_index=True)


def _stitch_fold_equity(parts: list[pd.Series]) -> pd.Series:
    stitched: list[pd.Series] = []
    chained = 1.0
    for equity in parts:
        clean = equity.dropna()
        if clean.empty:
            continue
        normalized = clean / float(clean.iloc[0]) * chained
        if stitched:
            normalized = normalized.iloc[1:]
        if normalized.empty:
            continue
        stitched.append(normalized)
        chained = float(normalized.iloc[-1])
    if not stitched:
        return pd.Series(dtype=float)
    out = pd.concat(stitched).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _basic_alpha_gate(metrics: Mapping[str, float], folds: pd.DataFrame) -> dict:
    returns = pd.to_numeric(folds["validation_return"], errors="coerce").dropna()
    gates = {
        "positive_oos_return": float(metrics["total_return"]) > 0.0,
        "sharpe_at_least_0_5": float(metrics["sharpe"]) >= 0.5,
        "max_drawdown_at_most_0_35": abs(float(metrics["max_drawdown"])) <= 0.35,
        "at_least_4_non_disastrous_folds": int((returns > -0.20).sum()) >= 4,
    }
    return {"passed": all(gates.values()), "gates": gates}


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    reference = ReferenceData.from_dir(args.reference_dir)
    audit = reference.audit()
    if audit.st_dates < audit.calendar_sessions or audit.limit_dates < audit.calendar_sessions:
        raise RuntimeError("strict PIT reference coverage is incomplete")
    universe = reference.codes_ever_active(args.start, args.end)
    if not universe:
        raise RuntimeError("point-in-time historical universe is empty")
    all_codes = list(dict.fromkeys(universe + [args.benchmark]))

    front_cache = Path(args.bar_cache_dir) / f"front_{args.data_start}_{args.end}"
    raw_cache = Path(args.bar_cache_dir) / f"none_limits_{args.data_start}_{args.end}"
    bars = load_daily_bars(all_codes, args.data_start, args.end, cache_dir=front_cache)
    raw_bars = load_limit_reference_bars(universe, args.data_start, args.end, cache_dir=raw_cache)
    adjusted_coverage = _coverage_or_fail("adjusted", universe, bars, args.min_symbol_coverage)
    raw_coverage = _coverage_or_fail("raw", universe, raw_bars, args.min_symbol_coverage)
    if args.benchmark not in bars or bars[args.benchmark].empty:
        raise RuntimeError("benchmark is missing")

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
    factor_cfg = V5FactorConfig()
    folds = nested_annual_folds(2021, 2025, outer_train_years=4, inner_validation_years=1)
    purged_folds = [purge_nested_fold(fold, calendar, max_forward_horizon=20) for fold in folds]

    sample_start_i = max(int(factor_cfg.warmup), int(calendar.searchsorted(pd.Timestamp(args.start), side="left")))
    sample_dates = calendar[sample_start_i:: args.rebalance_days]
    sample_dates = sample_dates[sample_dates <= pd.Timestamp(args.end)]
    target_dates = pd.DatetimeIndex(sample_dates)
    for row in purged_folds:
        for start, end in (
            (row.fold.inner_validation_start, row.fold.inner_validation_end),
            (row.fold.outer_validation_start, row.fold.outer_validation_end),
        ):
            target_dates = target_dates.union(
                _signal_dates_for_window(
                    bars,
                    args.benchmark,
                    strategy,
                    trade_start=start,
                    trade_end=min(end, pd.Timestamp(args.end)),
                )
            )
    target_dates = target_dates.sort_values().unique()

    stock_bars = {code: frame for code, frame in bars.items() if code != args.benchmark}
    close = _panel(stock_bars, "close", calendar)
    amount = _panel(stock_bars, "amount", calendar)
    suspend = _panel(stock_bars, "suspendFlag", calendar)
    raw_close = _panel(raw_bars, "close", calendar)
    benchmark_close = bars[args.benchmark]["close"].reindex(calendar).ffill()
    eligible = _eligible_mask(
        raw_close=raw_close,
        amount=amount,
        suspend=suspend,
        dates=target_dates,
        reference=reference,
        universe=universe,
        min_price=args.min_price,
        min_amount=args.min_amount,
        min_listing_sessions=args.min_listing_sessions,
        amount_window=factor_cfg.amount_window,
    )
    avg_amount = amount.rolling(factor_cfg.amount_window, min_periods=factor_cfg.amount_window).mean()
    liquidity = np.log1p(avg_amount.clip(lower=0.0)).reindex(target_dates).where(eligible)
    size_panel, size_meta = _load_size_panel(Path(args.exposure_root), target_dates, universe)
    size_panel = size_panel.where(eligible)
    snapshots = pd.read_parquet(args.industry_snapshots)
    industry = asof_industry_panel(snapshots, target_dates, universe).where(eligible)
    forward_20 = forward_return_panel(close, 20).reindex(sample_dates).where(eligible.reindex(sample_dates))

    raw_core: dict[str, pd.DataFrame] = {}
    for factor_name, raw_factor in iter_v5_raw_factors(close, amount, benchmark_close, factor_cfg):
        if factor_name not in CORE_ALPHA_FACTORS:
            del raw_factor
            continue
        print(f"C research factor: {factor_name}")
        raw_core[factor_name] = raw_factor.reindex(target_dates).where(eligible)
        del raw_factor
        gc.collect()
    missing_core = sorted(set(CORE_ALPHA_FACTORS).difference(raw_core))
    if missing_core:
        raise RuntimeError(f"missing core factors: {missing_core}")

    inputs = NeutralizationInputs(
        industry_panel=industry,
        size_panel=size_panel,
        liquidity_panel=liquidity,
    )
    variant_ranked: dict[str, dict[str, pd.DataFrame]] = {}
    variant_obs: dict[str, pd.DataFrame] = {}
    variant_meta: dict[str, dict] = {}
    for variant in VARIANTS:
        print(f"C neutralization variant: {variant}")
        neutralized = neutralize_factor_panels(
            raw_core,
            variant=variant,
            inputs=inputs,
            min_symbols=args.min_symbols_per_date,
            min_coverage=args.min_exposure_coverage,
        )
        variant_obs[variant] = _variant_observations(
            neutralized,
            forward_20,
            sample_dates,
            min_symbols=args.min_symbols_per_date,
        )
        variant_ranked[variant] = {
            factor: normalize_factor(
                panel,
                lower=factor_cfg.winsor_lower,
                upper=factor_cfg.winsor_upper,
            )
            for factor, panel in neutralized.items()
        }
        variant_meta[variant] = {
            "observation_rows": int(len(variant_obs[variant])),
            "factors": list(CORE_ALPHA_FACTORS),
        }
        del neutralized
        gc.collect()

    policy = CoreAlphaPolicy(include_challengers=False)
    outer_rows: list[dict] = []
    outer_equity: list[pd.Series] = []
    choice_rows: list[dict] = []
    selected_variants: list[str] = []

    for purged in purged_folds:
        fold = purged.fold
        inner_metrics: dict[str, Mapping[str, float]] = {}
        inner_payload: dict[str, dict] = {}
        for variant in VARIANTS:
            selection = select_core_alpha(
                variant_obs[variant],
                train_start=fold.inner_train_start,
                train_end=purged.inner_evidence_end,
                policy=policy,
            )
            score = apply_composite(variant_ranked[variant], selection.spec)
            result = _run_score(
                bars=bars,
                raw_bars=raw_bars,
                benchmark=args.benchmark,
                strategy=strategy,
                reference=reference,
                score=score,
                trade_start=fold.inner_validation_start,
                trade_end=fold.inner_validation_end,
                label=f"C inner {fold.outer_validation_year} {variant}",
            )
            inner_metrics[variant] = result.metrics
            inner_payload[variant] = {
                "selection": selection.to_dict(),
                "metrics": dict(result.metrics),
            }
        chosen = choose_inner_candidate(inner_metrics, primary_metric="sharpe", secondary_metric="total_return")
        selected_variants.append(chosen)
        outer_selection = select_core_alpha(
            variant_obs[chosen],
            train_start=fold.outer_train_start,
            train_end=purged.outer_evidence_end,
            policy=policy,
        )
        outer_score = apply_composite(variant_ranked[chosen], outer_selection.spec)
        outer = _run_score(
            bars=bars,
            raw_bars=raw_bars,
            benchmark=args.benchmark,
            strategy=strategy,
            reference=reference,
            score=outer_score,
            trade_start=fold.outer_validation_start,
            trade_end=fold.outer_validation_end,
            label=f"C outer {fold.outer_validation_year} {chosen}",
        )
        outer_equity.append(outer.equity["equity"].copy())
        outer_rows.append(
            {
                "validation_year": int(fold.outer_validation_year),
                "chosen_variant": chosen,
                "validation_return": float(outer.metrics["total_return"]),
                "sharpe": float(outer.metrics["sharpe"]),
                "max_drawdown": float(outer.metrics["max_drawdown"]),
                "trade_count": int(outer.metrics.get("trade_count", 0)),
                "selected_factors": ";".join(outer_selection.selected_factors),
                "weights_json": json.dumps(outer_selection.spec.weights, sort_keys=True),
                "inner_evidence_end": str(purged.inner_evidence_end.date()),
                "outer_evidence_end": str(purged.outer_evidence_end.date()),
            }
        )
        choice_rows.append(
            {
                "outer_validation_year": int(fold.outer_validation_year),
                "chosen_variant": chosen,
                "inner": inner_payload,
                "outer_selection": outer_selection.to_dict(),
            }
        )

    folds_frame = pd.DataFrame(outer_rows)
    folds_frame.to_csv(out / "nested_outer_folds.csv", index=False, encoding="utf-8-sig")
    _write_json(out / "nested_choices.json", choice_rows)
    stitched = _stitch_fold_equity(outer_equity)
    metrics = calculate_metrics(stitched) if not stitched.empty else {}
    metrics.update({
        "fold_count": int(len(folds_frame)),
        "positive_folds": int((folds_frame["validation_return"] > 0.0).sum()),
        "method": "purged_nested_core_alpha_neutralization_selection",
        "stock_selection_layer_only": True,
        "timing_override": "always_on",
    })
    gate = _basic_alpha_gate(metrics, folds_frame)
    _write_json(out / "nested_metrics.json", metrics)
    _write_json(out / "basic_alpha_gate.json", gate)

    mode_counts = Counter(selected_variants)
    final_variant = sorted(mode_counts, key=lambda name: (-mode_counts[name], name))[0]
    final_obs = variant_obs[final_variant]
    final_safe_end = pd.to_datetime(final_obs["date"], errors="coerce").max()
    if pd.isna(final_safe_end):
        raise RuntimeError("final candidate has no safe training observations")
    final_selection = select_core_alpha(
        final_obs,
        train_start=pd.Timestamp(args.data_start),
        train_end=final_safe_end,
        policy=policy,
    )
    candidate = FrozenCandidate(
        name="v5-c-core-neutralized",
        research_data_end="2025-12-31",
        neutralization_variant=final_variant,
        weights=final_selection.spec.weights,
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        execution_delay_sessions=strategy.execution_delay_sessions,
        min_price=args.min_price,
        min_amount=args.min_amount,
        min_listing_sessions=args.min_listing_sessions,
    )
    candidate_payload = {
        "basic_alpha_gate_passed": bool(gate["passed"]),
        "variant_choice_rule": "mode_of_inner-validation choices; lexical tie-break",
        "variant_counts": dict(mode_counts),
        "final_training_evidence_end": str(pd.Timestamp(final_safe_end).date()),
        "selection": final_selection.to_dict(),
        "frozen": freeze_candidate_manifest(candidate),
    }
    _write_json(out / "candidate_manifest.json", candidate_payload)
    _write_json(
        out / "research_manifest.json",
        {
            "adjusted_symbol_coverage": adjusted_coverage,
            "raw_symbol_coverage": raw_coverage,
            "size_exposure": size_meta,
            "industry_snapshot_rows": int(len(snapshots)),
            "target_dates": int(len(target_dates)),
            "sample_dates": int(len(sample_dates)),
            "variants": variant_meta,
            "basic_alpha_gate_passed": bool(gate["passed"]),
            "candidate_sha256": candidate.fingerprint(),
            "holdout_unlocked": bool(gate["passed"]),
        },
    )
    print(json.dumps({"metrics": metrics, "gate": gate, "candidate": candidate_payload}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
