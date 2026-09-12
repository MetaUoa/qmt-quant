from pathlib import Path

from qmt_quant.workflow_contract import load_workflow, normalized_run


WORKFLOW = Path(".github/workflows/tests.yml")


def test_targeted_mypy_covers_expanded_core_safety_modules() -> None:
    workflow = load_workflow(WORKFLOW)
    command = normalized_run(workflow, "offline-tests", "Mypy safety-contract gate")
    for path in (
        "qmt_quant/production_candidate.py",
        "qmt_quant/acceptance_lineage.py",
        "qmt_quant/run_manifest.py",
        "qmt_quant/cost_manifest.py",
        "qmt_quant/historical_fees.py",
        "qmt_quant/raw_ledger.py",
        "qmt_quant/live_safety.py",
        "qmt_quant/live_trader.py",
        "qmt_quant/broker_events.py",
        "qmt_quant/freshness.py",
        "qmt_quant/transaction_costs.py",
        "qmt_quant/target_planning.py",
        "qmt_quant/execution_state.py",
        "qmt_quant/execution_phases.py",
        "qmt_quant/execution_recovery.py",
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
        "qmt_quant/factor_orthogonality.py",
        "qmt_quant/composites.py",
        "qmt_quant/neutralization_diagnostics.py",
        "qmt_quant/v5_selector.py",
        "generate_live_targets.py",
        "build_cost_manifest.py",
        "run_acceptance.py",
        "run_v5_c9_neutralization_diagnostics.py",
        "run_qmt_executor.py",
        "reconcile_qmt_execution.py",
        "risk/pretrade.py",
        "risk/runtime.py",
        "monitoring/check_runtime.py",
        "monitoring/alerts.py",
    ):
        assert path in command
    assert "--ignore-missing-imports" in command
    assert "--check-untyped-defs" in command
    assert "--explicit-package-bases" in command
