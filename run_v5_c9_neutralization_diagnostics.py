from __future__ import annotations

from collections.abc import Callable
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol, cast

import pandas as pd

from qmt_quant.nested_walk_forward import PurgedNestedFold
from qmt_quant.neutralization_diagnostics import (
    aggregate_variant_quality,
    summarize_neutralization_variants,
)
from qmt_quant.research_policy import (
    DEFAULT_RESEARCH_DATA_POLICY,
    assert_cli_float_floor,
    assert_cli_int_floor,
    assert_pre_holdout_end,
    cli_value,
)
from qmt_quant.research_runtime import install_v5_c_contracts


class _C1Surface(Protocol):
    VARIANTS: tuple[str, ...]
    CORE_ALPHA_FACTORS: tuple[str, ...]
    _variant_observations: Callable[..., pd.DataFrame]
    purge_nested_fold: Callable[..., PurgedNestedFold]
    nested_annual_folds: Callable[..., list[Any]]
    main: Callable[[], int]


_c1_module = importlib.import_module("run_v5_c_nested_research")
c1 = cast(_C1Surface, _c1_module)

_OriginalVariantObservations = c1._variant_observations
_OriginalPurgeNestedFold = c1.purge_nested_fold
_C1_CONTRACT_HOOK_NAMES = (
    "_coverage_or_fail",
    "_eligible_mask",
    "_assert_strict_metrics",
    "_stitch_fold_equity",
    "_basic_alpha_gate",
)
_CAPTURED: list[pd.DataFrame] = []
_CAPTURED_FOLDS: list[PurgedNestedFold] = []
_FORBIDDEN_C1_OUTPUTS = (
    "candidate_manifest.json",
    "basic_alpha_gate.json",
    "research_manifest.json",
    "nested_choices.json",
    "nested_metrics.json",
    "nested_outer_folds.csv",
)


class _C9DiagnosticsReady(Exception):
    """Internal sentinel: all C9 diagnostic inputs are captured; stop C1 immediately."""


def _arg_value(argv: list[str], name: str) -> str | None:
    return cli_value(argv, name)


def _assert_pre_2026_only(argv: list[str]) -> None:
    assert_pre_holdout_end(argv, context="C9 research")


def _assert_data_policy(argv: list[str]) -> None:
    policy = DEFAULT_RESEARCH_DATA_POLICY
    assert_cli_float_floor(
        argv,
        "--min-symbol-coverage",
        minimum=policy.min_symbol_coverage,
        default=policy.min_symbol_coverage,
    )
    assert_cli_float_floor(
        argv,
        "--min-exposure-coverage",
        minimum=policy.min_exposure_coverage,
        default=policy.min_exposure_coverage,
    )
    assert_cli_int_floor(
        argv,
        "--min-symbols-per-date",
        minimum=policy.min_symbols_per_date,
        default=policy.min_symbols_per_date,
    )


def _capture_purged_fold(*args, **kwargs) -> PurgedNestedFold:
    purged = _OriginalPurgeNestedFold(*args, **kwargs)
    _CAPTURED_FOLDS.append(purged)
    return purged


def _capture_variant_observations(*args, **kwargs) -> pd.DataFrame:
    frame = _OriginalVariantObservations(*args, **kwargs)
    index = len(_CAPTURED)
    if index >= len(c1.VARIANTS):
        raise RuntimeError("C9 observed more neutralization variants than declared")
    tagged = frame.copy()
    tagged.insert(0, "variant", c1.VARIANTS[index])
    _CAPTURED.append(tagged)
    if len(_CAPTURED) == len(c1.VARIANTS):
        raise _C9DiagnosticsReady
    return frame


def _build_fold_safe_diagnostics(output: Path) -> dict[str, Any]:
    if len(_CAPTURED) != len(c1.VARIANTS):
        raise RuntimeError(
            f"C9 expected exactly {len(c1.VARIANTS)} captured variants, found {len(_CAPTURED)}"
        )
    expected_folds = len(
        c1.nested_annual_folds(
            2021,
            2025,
            outer_train_years=4,
            inner_validation_years=1,
        )
    )
    if len(_CAPTURED_FOLDS) != expected_folds:
        raise RuntimeError(
            f"C9 expected exactly {expected_folds} purged folds, found {len(_CAPTURED_FOLDS)}"
        )

    observations = pd.concat(_CAPTURED, ignore_index=True)
    factor_rows: list[pd.DataFrame] = []
    quality_rows: list[pd.DataFrame] = []
    windows: list[dict[str, Any]] = []
    for purged in _CAPTURED_FOLDS:
        year = int(purged.fold.outer_validation_year)
        for phase, start, end in (
            (
                "inner",
                purged.fold.inner_train_start,
                purged.inner_evidence_end,
            ),
            (
                "outer",
                purged.fold.outer_train_start,
                purged.outer_evidence_end,
            ),
        ):
            start_text = str(pd.Timestamp(start).date())
            end_text = str(pd.Timestamp(end).date())
            if pd.Timestamp(end_text) >= pd.Timestamp("2026-01-01"):
                raise RuntimeError("C9 diagnostic window crossed into 2026")
            summary = summarize_neutralization_variants(
                observations,
                start=start_text,
                end=end_text,
            )
            summary.insert(0, "validation_year", year)
            summary.insert(1, "phase", phase)
            factor_rows.append(summary)

            quality = aggregate_variant_quality(
                summary.drop(columns=["validation_year", "phase"])
            )
            quality.insert(0, "validation_year", year)
            quality.insert(1, "phase", phase)
            quality_rows.append(quality)
            windows.append(
                {
                    "validation_year": year,
                    "phase": phase,
                    "train_start": start_text,
                    "train_end": end_text,
                }
            )

    factor_summary = pd.concat(factor_rows, ignore_index=True)
    variant_quality = pd.concat(quality_rows, ignore_index=True)
    factor_summary.to_csv(
        output / "c9_neutralization_factor_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    variant_quality.to_csv(
        output / "c9_neutralization_variant_quality.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload: dict[str, Any] = {
        "method": "fold_safe_neutralization_diagnostics_only",
        "selection_changed": False,
        "winner_selection_executed": False,
        "candidate_changed": False,
        "candidate_manifest_written": False,
        "basic_alpha_gate_evaluated": False,
        "holdout_unlocked": False,
        "pre_2026_only": True,
        "canonical_c1_contracts": True,
        "core_factors_only": list(c1.CORE_ALPHA_FACTORS),
        "variants": list(c1.VARIANTS),
        "captured_variant_count": len(_CAPTURED),
        "fold_count": len(_CAPTURED_FOLDS),
        "windows": windows,
    }
    (output / "c9_diagnostics_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _assert_output_namespace_safe(output: Path) -> None:
    conflicts = [name for name in _FORBIDDEN_C1_OUTPUTS if (output / name).exists()]
    if conflicts:
        raise RuntimeError(
            "C9 diagnostics refuse to delete or overwrite C1 selection artifacts; "
            "use a dedicated output directory. Conflicts: " + ", ".join(conflicts)
        )


def _assert_no_forbidden_c1_outputs(output: Path) -> None:
    leaked = [name for name in _FORBIDDEN_C1_OUTPUTS if (output / name).exists()]
    if leaked:
        raise RuntimeError(f"C9 produced forbidden C1 selection outputs: {', '.join(leaked)}")


def main() -> int:
    argv = sys.argv[1:]
    _assert_pre_2026_only(argv)
    _assert_data_policy(argv)
    output = Path(_arg_value(argv, "--output") or "output/v5_c_nested")
    output.mkdir(parents=True, exist_ok=True)
    _assert_output_namespace_safe(output)
    _CAPTURED.clear()
    _CAPTURED_FOLDS.clear()

    previous_variant_observations = c1._variant_observations
    previous_purge_nested_fold = c1.purge_nested_fold
    previous_contract_hooks = {
        name: getattr(_c1_module, name)
        for name in _C1_CONTRACT_HOOK_NAMES
    }
    install_v5_c_contracts(_c1_module)
    c1._variant_observations = _capture_variant_observations
    c1.purge_nested_fold = _capture_purged_fold
    try:
        try:
            c1.main()
        except _C9DiagnosticsReady:
            pass
        else:
            raise RuntimeError(
                "C9 reached the C1 selection path before diagnostics capture stopped it"
            )
    finally:
        c1._variant_observations = previous_variant_observations
        c1.purge_nested_fold = previous_purge_nested_fold
        for name, value in previous_contract_hooks.items():
            setattr(_c1_module, name, value)

    _build_fold_safe_diagnostics(output)
    _assert_no_forbidden_c1_outputs(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
