from pathlib import Path

from qmt_quant.workflow_contract import load_workflow, normalized_run


WORKFLOW = Path(".github/workflows/tests.yml")


def test_targeted_mypy_covers_expanded_core_safety_modules() -> None:
    workflow = load_workflow(WORKFLOW)
    command = normalized_run(workflow, "offline-tests", "Mypy safety-contract gate")
    for path in (
        "qmt_quant/production_candidate.py",
        "qmt_quant/v5_gates.py",
        "qmt_quant/adjustment_provenance.py",
        "qmt_quant/backtest.py",
        "qmt_quant/backtest_execution.py",
        "qmt_quant/backtest_reporting.py",
        "qmt_quant/backtest_selection.py",
        "qmt_quant/backtest_sell_execution.py",
        "qmt_quant/backtest_buy_execution.py",
        "qmt_quant/backtest_trades.py",
        "qmt_quant/backtest_equity.py",
        "qmt_quant/reference_data.py",
        "qmt_quant/research_policy.py",
        "qmt_quant/research_contracts.py",
        "qmt_quant/research_runtime.py",
        "qmt_quant/factor_diagnostics.py",
        "qmt_quant/factor_selection.py",
        "qmt_quant/v5_selector.py",
        "risk/runtime.py",
        "monitoring/alerts.py",
    ):
        assert path in command
    assert "--ignore-missing-imports" in command
    assert "--check-untyped-defs" in command
