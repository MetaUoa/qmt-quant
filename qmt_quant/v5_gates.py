from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V5GateConfig:
    """Research gates for advancing a V5 candidate beyond alpha discovery."""

    min_oos_total_return: float = 0.0
    min_oos_sharpe: float = 0.50
    max_oos_drawdown: float = 0.35
    min_non_disastrous_folds: int = 4
    disastrous_fold_loss: float = -0.20
    min_stress_pass_ratio: float = 0.75
    min_robustness_pass_ratio: float = 0.67


def _gate(name: str, value, threshold: str, passed: bool) -> dict:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
    }


def evaluate_v5_gates(
    metrics: dict,
    folds: pd.DataFrame,
    *,
    stress_pass_ratio: float | None = None,
    robustness_pass_ratio: float | None = None,
    config: V5GateConfig | None = None,
) -> dict:
    """Evaluate V5 promotion gates without relaxing any backtest/data checks.

    Missing stress or robustness evidence is intentionally fail-closed: a candidate
    cannot be promoted merely because those experiments have not been run yet.
    """
    cfg = config or V5GateConfig()
    total_return = float(metrics.get("total_return", np.nan))
    sharpe = float(metrics.get("sharpe", np.nan))
    drawdown = abs(float(metrics.get("max_drawdown", np.nan)))

    fold_returns = pd.to_numeric(
        folds.get("validation_return", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    non_disastrous = int((fold_returns > cfg.disastrous_fold_loss).sum())

    gates = [
        _gate(
            "positive_oos_return",
            total_return,
            f"> {cfg.min_oos_total_return:.3f}",
            np.isfinite(total_return) and total_return > cfg.min_oos_total_return,
        ),
        _gate(
            "oos_sharpe",
            sharpe,
            f">= {cfg.min_oos_sharpe:.2f}",
            np.isfinite(sharpe) and sharpe >= cfg.min_oos_sharpe,
        ),
        _gate(
            "oos_max_drawdown",
            drawdown,
            f"<= {cfg.max_oos_drawdown:.2f}",
            np.isfinite(drawdown) and drawdown <= cfg.max_oos_drawdown,
        ),
        _gate(
            "non_disastrous_oos_folds",
            non_disastrous,
            f">= {cfg.min_non_disastrous_folds} folds with return > {cfg.disastrous_fold_loss:.2f}",
            len(fold_returns) >= cfg.min_non_disastrous_folds
            and non_disastrous >= cfg.min_non_disastrous_folds,
        ),
        _gate(
            "stress_resilience",
            stress_pass_ratio,
            f">= {cfg.min_stress_pass_ratio:.2f}",
            stress_pass_ratio is not None
            and np.isfinite(float(stress_pass_ratio))
            and float(stress_pass_ratio) >= cfg.min_stress_pass_ratio,
        ),
        _gate(
            "parameter_robustness",
            robustness_pass_ratio,
            f">= {cfg.min_robustness_pass_ratio:.2f}",
            robustness_pass_ratio is not None
            and np.isfinite(float(robustness_pass_ratio))
            and float(robustness_pass_ratio) >= cfg.min_robustness_pass_ratio,
        ),
    ]
    return {
        "passed": all(row["passed"] for row in gates),
        "gates": gates,
        "fold_count": int(len(fold_returns)),
        "non_disastrous_folds": non_disastrous,
    }
