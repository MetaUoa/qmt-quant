import pandas as pd

from qmt_quant.v5_gates import evaluate_v5_gates


def test_v5_gates_pass_when_all_evidence_is_strong():
    metrics = {"total_return": 0.35, "sharpe": 0.9, "max_drawdown": -0.22}
    folds = pd.DataFrame({"validation_return": [0.08, 0.04, -0.05, 0.12, 0.03]})
    result = evaluate_v5_gates(
        metrics,
        folds,
        stress_pass_ratio=0.80,
        robustness_pass_ratio=0.75,
    )
    assert result["passed"] is True
    assert all(row["passed"] for row in result["gates"])


def test_v5_gates_fail_closed_when_stress_or_robustness_missing():
    metrics = {"total_return": 0.35, "sharpe": 0.9, "max_drawdown": -0.22}
    folds = pd.DataFrame({"validation_return": [0.08, 0.04, -0.05, 0.12, 0.03]})
    result = evaluate_v5_gates(metrics, folds)
    by_name = {row["name"]: row for row in result["gates"]}
    assert result["passed"] is False
    assert by_name["stress_resilience"]["passed"] is False
    assert by_name["parameter_robustness"]["passed"] is False


def test_v5_gates_reject_large_oos_drawdown_and_repeated_bad_folds():
    metrics = {"total_return": 0.10, "sharpe": 0.7, "max_drawdown": -0.50}
    folds = pd.DataFrame({"validation_return": [-0.30, -0.25, 0.10, -0.22, 0.15]})
    result = evaluate_v5_gates(
        metrics,
        folds,
        stress_pass_ratio=0.90,
        robustness_pass_ratio=0.90,
    )
    by_name = {row["name"]: row for row in result["gates"]}
    assert by_name["oos_max_drawdown"]["passed"] is False
    assert by_name["non_disastrous_oos_folds"]["passed"] is False
    assert result["passed"] is False
