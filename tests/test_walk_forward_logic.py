from __future__ import annotations

import math

import pytest

from run_walk_forward import _floats, _ints, _train_score


def test_grid_parsers():
    assert _ints("3, 5,10") == [3, 5, 10]
    assert _floats("0, 0.02") == [0.0, 0.02]


def test_train_score_rejects_excessive_drawdown():
    metrics = {"max_drawdown": -0.51, "calmar": 5.0, "sharpe": 3.0, "cagr": 2.0}
    assert _train_score(metrics, 0.50) == float("-inf")


def test_train_score_prefers_risk_adjusted_candidate():
    safer = {"max_drawdown": -0.20, "calmar": 2.0, "sharpe": 1.5, "cagr": 0.40}
    weaker = {"max_drawdown": -0.20, "calmar": 1.0, "sharpe": 0.8, "cagr": 0.60}
    assert math.isfinite(_train_score(safer, 0.50))
    assert _train_score(safer, 0.50) > _train_score(weaker, 0.50)
