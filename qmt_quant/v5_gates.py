from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, cast

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


FROZEN_V5_GATE_CONFIG = V5GateConfig()


def _finite_gate_value(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{name} must be finite, got {numeric!r}")
    return numeric


def assert_v5_gate_config(config: V5GateConfig | None = None) -> V5GateConfig:
    """Reject custom gate configurations that loosen the frozen V5 policy.

    Stricter research gates remain valid. The historical default values are policy
    invariants rather than tuning knobs, so NaN/Inf and any relaxation fail closed.
    """
    cfg = config or FROZEN_V5_GATE_CONFIG
    frozen = FROZEN_V5_GATE_CONFIG

    lower_bounds = (
        ("min_oos_total_return", cfg.min_oos_total_return, frozen.min_oos_total_return),
        ("min_oos_sharpe", cfg.min_oos_sharpe, frozen.min_oos_sharpe),
        ("disastrous_fold_loss", cfg.disastrous_fold_loss, frozen.disastrous_fold_loss),
        ("min_stress_pass_ratio", cfg.min_stress_pass_ratio, frozen.min_stress_pass_ratio),
        (
            "min_robustness_pass_ratio",
            cfg.min_robustness_pass_ratio,
            frozen.min_robustness_pass_ratio,
        ),
    )
    for name, value, minimum in lower_bounds:
        numeric = _finite_gate_value(value, name)
        if numeric < float(minimum):
            raise RuntimeError(
                f"{name}={numeric:g} loosens frozen V5 minimum {float(minimum):g}"
            )

    drawdown = _finite_gate_value(cfg.max_oos_drawdown, "max_oos_drawdown")
    if drawdown > frozen.max_oos_drawdown:
        raise RuntimeError(
            f"max_oos_drawdown={drawdown:g} loosens frozen V5 maximum "
            f"{frozen.max_oos_drawdown:g}"
        )

    fold_count = int(cfg.min_non_disastrous_folds)
    if fold_count < frozen.min_non_disastrous_folds:
        raise RuntimeError(
            f"min_non_disastrous_folds={fold_count} loosens frozen V5 minimum "
            f"{frozen.min_non_disastrous_folds}"
        )
    return cfg


def _gate(name: str, value, threshold: str, passed: bool) -> dict:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _metric_float(value: object) -> float:
    """Convert report payload values without weakening their external mapping type."""
    return float(cast(Any, value))


def evaluate_basic_alpha_gate(
    metrics: Mapping[str, object],
    folds: pd.DataFrame,
    *,
    config: V5GateConfig | None = None,
) -> dict:
    """Evaluate the exact C1/C7 pre-holdout Basic Alpha Gate.

    The return shape intentionally matches the historical nested-research JSON so
    callers can centralize policy without changing frozen report semantics.
    """
    cfg = assert_v5_gate_config(config)
    returns = pd.to_numeric(
        folds.get("validation_return", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    total_return = _metric_float(metrics.get("total_return", np.nan))
    sharpe = _metric_float(metrics.get("sharpe", np.nan))
    drawdown = abs(_metric_float(metrics.get("max_drawdown", np.nan)))
    gates = {
        "positive_oos_return": (
            np.isfinite(total_return) and total_return > cfg.min_oos_total_return
        ),
        "sharpe_at_least_0_5": (
            np.isfinite(sharpe) and sharpe >= cfg.min_oos_sharpe
        ),
        "max_drawdown_at_most_0_35": (
            np.isfinite(drawdown) and drawdown <= cfg.max_oos_drawdown
        ),
        "at_least_4_non_disastrous_folds": int(
            (returns > cfg.disastrous_fold_loss).sum()
        )
        >= cfg.min_non_disastrous_folds,
    }
    return {"passed": all(gates.values()), "gates": gates}


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
    cfg = assert_v5_gate_config(config)
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
