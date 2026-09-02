from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .config import AcceptanceConfig


def _fold_stats(folds: pd.DataFrame | None) -> tuple[int, int]:
    if folds is None or folds.empty or "validation_return" not in folds:
        return 0, 0
    vals = pd.to_numeric(folds["validation_return"], errors="coerce").dropna()
    return int((vals > 0).sum()), int(len(vals))


def grade_strategy(
    backtest_metrics: dict,
    oos_metrics: dict,
    folds: pd.DataFrame | None,
    stress: dict,
    cfg: AcceptanceConfig | None = None,
) -> dict:
    cfg = cfg or AcceptanceConfig()
    multiple = float(backtest_metrics.get("multiple", 0.0) or 0.0)
    max_dd = abs(float(backtest_metrics.get("max_drawdown", -1.0) or -1.0))
    sharpe = float(backtest_metrics.get("sharpe", 0.0) or 0.0)
    oos_cagr = float(oos_metrics.get("cagr", 0.0) or 0.0)
    positive_folds, fold_count = _fold_stats(folds)
    pass_ratio = float(stress.get("pass_ratio", 0.0) or 0.0)

    a_checks = {
        "multiple": multiple >= cfg.target_multiple,
        "max_drawdown": max_dd <= cfg.max_drawdown_a,
        "sharpe": sharpe >= cfg.min_sharpe_a,
        "oos_cagr": oos_cagr >= cfg.min_oos_cagr_a,
        "positive_oos_folds": positive_folds >= cfg.min_positive_oos_folds_a,
        "stress_pass_ratio": pass_ratio >= cfg.min_stress_pass_ratio_a,
    }
    b_checks = {
        "multiple": multiple >= cfg.grade_b_multiple,
        "max_drawdown": max_dd <= cfg.max_drawdown_b,
        "sharpe": sharpe >= cfg.min_sharpe_b,
        "oos_cagr": oos_cagr >= cfg.min_oos_cagr_b,
        "positive_oos_folds": positive_folds >= cfg.min_positive_oos_folds_b,
        "stress_pass_ratio": pass_ratio >= cfg.min_stress_pass_ratio_b,
    }
    c_checks = {
        "profitable_full_period": multiple > 1.0,
        "profitable_oos": oos_cagr > 0.0,
        "oos_majority_positive": positive_folds >= max(1, (fold_count + 1) // 2),
        "stress_majority_pass": pass_ratio >= 0.50,
    }

    if all(a_checks.values()):
        grade = "A"
        status = "150x target passed with OOS/risk gates"
    elif all(b_checks.values()):
        grade = "B"
        status = "50x+ robust candidate; 150x gate not fully satisfied"
    elif all(c_checks.values()):
        grade = "C"
        status = "positive/replicable candidate; return target not reached"
    else:
        grade = "REJECT"
        status = "failed robustness or profitability gates"

    return {
        "grade": grade,
        "status": status,
        "observed": {
            "multiple": multiple,
            "max_drawdown_abs": max_dd,
            "sharpe": sharpe,
            "oos_cagr": oos_cagr,
            "positive_oos_folds": positive_folds,
            "oos_fold_count": fold_count,
            "stress_pass_ratio": pass_ratio,
        },
        "grade_a_checks": a_checks,
        "grade_b_checks": b_checks,
        "grade_c_checks": c_checks,
        "thresholds": asdict(cfg),
    }
