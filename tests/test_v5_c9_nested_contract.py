from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

import run_v5_c9_neutralization_diagnostics as c9
from qmt_quant.workflow_contract import (
    env_value,
    load_workflow,
    normalized_run,
    workflow_events,
)


WORKFLOW = Path(".github/workflows/v5-c9-neutralization-diagnostics.yml")


def test_c9_rejects_holdout_dates_and_cli_bypasses() -> None:
    c9._assert_pre_2026_only(["--end", "20251231"])
    c9._assert_pre_2026_only(["--end=20251231"])
    with pytest.raises(RuntimeError, match="holdout remains blinded"):
        c9._assert_pre_2026_only(["--end", "20260101"])
    with pytest.raises(RuntimeError, match="holdout remains blinded"):
        c9._assert_pre_2026_only(["--end=20260101"])
    with pytest.raises(RuntimeError, match="duplicate --end"):
        c9._assert_pre_2026_only(["--end", "20251231", "--end=20260101"])


def test_c9_workflow_is_manual_only_and_pinned_to_authoritative_history() -> None:
    workflow = load_workflow(WORKFLOW)
    events = workflow_events(workflow)
    assert set(events) == {"workflow_dispatch"}
    assert env_value(workflow, "SHARD_COUNT") == "20"
    assert env_value(workflow, "SOURCE_RUN_ID") == "33811845110"
    assert env_value(workflow, "RECOVERY_RUN_ID") == "33887254974"
    assert env_value(workflow, "EXPOSURE_RUN_ID") == "33963211771"
    assert env_value(workflow, "INDUSTRY_RUN_ID") == "33969253365"
    assert env_value(workflow, "QMT_QUANT_CACHE_ONLY") == "1"
    audit = normalized_run(
        workflow,
        "diagnostics",
        "Revalidate full historical data before C9 diagnostics",
    )
    runner = normalized_run(
        workflow,
        "diagnostics",
        "Run strict purged C1 nested path with fold-safe C9 diagnostics",
    )
    install = normalized_run(workflow, "diagnostics", "Install research dependencies")
    assert "python -m pip install -r requirements.txt" in install
    assert "python -m pip check" in install
    assert "--min-symbol-coverage 0.98" in audit
    assert "--min-session-coverage 0.97" in audit
    assert "--end 20251231" in runner
    assert "--min-exposure-coverage 0.95" in runner


def _capture_sample_observations() -> None:
    c9._CAPTURED.clear()
    dates = pd.to_datetime(
        [
            "2018-01-02",
            "2019-01-02",
            "2020-01-02",
            "2021-01-04",
            "2022-01-04",
            "2023-01-03",
            "2024-01-02",
        ]
    )
    for variant in c9.c1.VARIANTS:
        rows = []
        for factor in c9.c1.CORE_ALPHA_FACTORS:
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "variant": variant,
                        "factor": factor,
                        "date": date,
                        "rank_ic": 0.01 * (i + 1),
                    }
                )
        c9._CAPTURED.append(pd.DataFrame(rows))


def _capture_sample_purged_folds() -> None:
    c9._CAPTURED_FOLDS.clear()
    calendar = pd.bdate_range("2017-01-02", "2025-12-31")
    folds = c9.c1.nested_annual_folds(
        2021,
        2025,
        outer_train_years=4,
        inner_validation_years=1,
    )
    c9._CAPTURED_FOLDS.extend(
        c9.c1.purge_nested_fold(fold, calendar, max_forward_horizon=20)
        for fold in folds
    )


def test_c9_fold_safe_diagnostics_do_not_change_selection(tmp_path: Path) -> None:
    _capture_sample_observations()
    _capture_sample_purged_folds()

    manifest = c9._build_fold_safe_diagnostics(tmp_path)
    assert manifest["selection_changed"] is False
    assert manifest["winner_selection_executed"] is False
    assert manifest["candidate_changed"] is False
    assert manifest["candidate_manifest_written"] is False
    assert manifest["basic_alpha_gate_evaluated"] is False
    assert manifest["holdout_unlocked"] is False
    assert manifest["pre_2026_only"] is True
    assert manifest["canonical_c1_contracts"] is True
    assert manifest["core_factors_only"] == list(c9.c1.CORE_ALPHA_FACTORS)
    assert manifest["fold_count"] == 5
    assert len(manifest["windows"]) == 10
    assert (tmp_path / "c9_neutralization_factor_summary.csv").exists()
    assert (tmp_path / "c9_neutralization_variant_quality.csv").exists()
    assert not (tmp_path / "candidate_manifest.json").exists()
    assert not (tmp_path / "basic_alpha_gate.json").exists()
    assert not (tmp_path / "research_manifest.json").exists()


def test_c9_main_stops_immediately_after_fourth_variant_capture(monkeypatch, tmp_path: Path) -> None:
    previous_variant_observations = c9.c1._variant_observations
    previous_purge_nested_fold = c9.c1.purge_nested_fold
    previous_contract_hooks = {
        name: getattr(c9.c1, name)
        for name in c9._C1_CONTRACT_HOOK_NAMES
    }
    winner_reached = False
    diagnostics_built = False
    installed_marker = object()

    def fake_variant_observations(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "factor": [c9.c1.CORE_ALPHA_FACTORS[0]],
                "horizon": [20],
                "date": [pd.Timestamp("2020-01-02")],
                "rank_ic": [0.01],
            }
        )

    def fake_install(module) -> None:
        for name in c9._C1_CONTRACT_HOOK_NAMES:
            setattr(module, name, installed_marker)

    def fake_c1_main() -> int:
        nonlocal winner_reached
        for _ in c9.c1.VARIANTS:
            c9.c1._variant_observations(None, None, None, min_symbols=50)
        winner_reached = True
        return 0

    def fake_build(output: Path) -> dict:
        nonlocal diagnostics_built
        diagnostics_built = True
        assert output == tmp_path
        assert len(c9._CAPTURED) == len(c9.c1.VARIANTS)
        return {"winner_selection_executed": False}

    monkeypatch.setattr(c9, "_OriginalVariantObservations", fake_variant_observations)
    monkeypatch.setattr(c9, "install_v5_c_contracts", fake_install)
    monkeypatch.setattr(c9.c1, "main", fake_c1_main)
    monkeypatch.setattr(c9, "_build_fold_safe_diagnostics", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_v5_c9_neutralization_diagnostics.py", "--output", str(tmp_path)],
    )

    assert c9.main() == 0
    assert winner_reached is False
    assert diagnostics_built is True
    assert c9.c1._variant_observations is previous_variant_observations
    assert c9.c1.purge_nested_fold is previous_purge_nested_fold
    for name, value in previous_contract_hooks.items():
        assert getattr(c9.c1, name) is value


def test_c9_refuses_existing_selection_outputs_without_deleting_them(tmp_path: Path) -> None:
    existing = tmp_path / "candidate_manifest.json"
    existing.write_text("frozen", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refuse to delete or overwrite"):
        c9._assert_output_namespace_safe(tmp_path)
    assert existing.read_text(encoding="utf-8") == "frozen"
