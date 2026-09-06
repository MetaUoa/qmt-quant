from __future__ import annotations

from inspect import signature
from pathlib import Path

from qmt_quant.research_policy import DEFAULT_V5_SELECTION_POLICY, V5SelectionPolicy
from qmt_quant.v5_selector import DEFAULT_SAFE_FACTORS, select_training_composite
from qmt_quant.workflow_contract import load_workflow, normalized_run


ROOT = Path(__file__).resolve().parents[1]


def test_v5_selection_policy_preserves_existing_historical_defaults() -> None:
    policy = DEFAULT_V5_SELECTION_POLICY
    assert policy == V5SelectionPolicy()
    assert policy.safe_factors == (
        "low_volatility",
        "low_downside_risk",
        "liquidity_stability",
        "short_reversal",
        "momentum_20_5",
        "momentum_60_5",
        "momentum_120_5",
        "trend_quality",
        "trend_persistence",
    )
    assert policy.correlation_horizon == 20
    assert policy.min_abs_rank_ic == 0.01
    assert policy.max_abs_correlation == 0.80
    assert policy.min_factors == 2
    assert policy.max_factors == 4
    assert policy.weight_metric_cap == 0.10
    assert DEFAULT_SAFE_FACTORS == policy.safe_factors


def test_selector_signature_sources_all_defaults_from_frozen_policy() -> None:
    params = signature(select_training_composite).parameters
    policy = DEFAULT_V5_SELECTION_POLICY
    assert params["allowed_factors"].default == policy.safe_factors
    assert params["correlation_horizon"].default == policy.correlation_horizon
    assert params["min_abs_rank_ic"].default == policy.min_abs_rank_ic
    assert params["max_abs_correlation"].default == policy.max_abs_correlation
    assert params["min_factors"].default == policy.min_factors
    assert params["max_factors"].default == policy.max_factors
    assert params["weight_metric_cap"].default == policy.weight_metric_cap


def test_v5_selector_is_in_targeted_mypy_gate() -> None:
    workflow = load_workflow(ROOT / ".github/workflows/tests.yml")
    command = normalized_run(workflow, "offline-tests", "Mypy safety-contract gate")
    assert "qmt_quant/v5_selector.py" in command
