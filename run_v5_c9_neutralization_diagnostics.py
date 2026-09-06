from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

import run_v5_c_nested_research as c1
from qmt_quant.neutralization_diagnostics import (
    aggregate_variant_quality,
    summarize_neutralization_variants,
)
from qmt_quant.research_runtime import install_v5_c_contracts


_MAX_RESEARCH_END = "20251231"
_OriginalVariantObservations = c1._variant_observations
_CAPTURED: list[pd.DataFrame] = []


def _arg_value(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        raise RuntimeError(f"missing value for {name}")
    return argv[index + 1]


def _assert_pre_2026_only(argv: list[str]) -> None:
    end = (_arg_value(argv, "--end") or _MAX_RESEARCH_END).replace("-", "")
    if not end.isdigit() or len(end) != 8:
        raise RuntimeError("C9 research end must be YYYYMMDD or YYYY-MM-DD")
    if end > _MAX_RESEARCH_END:
        raise RuntimeError("C9 is pre-2026 research only; holdout remains blinded")


def _capture_variant_observations(*args, **kwargs) -> pd.DataFrame:
    frame = _OriginalVariantObservations(*args, **kwargs)
    index = len(_CAPTURED)
    if index >= len(c1.VARIANTS):
        raise RuntimeError("C9 observed more neutralization variants than declared")
    tagged = frame.copy()
    tagged.insert(0, "variant", c1.VARIANTS[index])
    _CAPTURED.append(tagged)
    return frame


def _build_fold_safe_diagnostics(output: Path) -> dict:
    if len(_CAPTURED) != len(c1.VARIANTS):
        raise RuntimeError(
            f"C9 expected exactly {len(c1.VARIANTS)} captured variants, found {len(_CAPTURED)}"
        )
    choices_path = output / "nested_choices.json"
    if not choices_path.exists():
        raise RuntimeError("C9 requires nested_choices.json from the strict C1 nested run")
    choices = json.loads(choices_path.read_text(encoding="utf-8"))
    observations = pd.concat(_CAPTURED, ignore_index=True)

    factor_rows: list[pd.DataFrame] = []
    quality_rows: list[pd.DataFrame] = []
    windows: list[dict] = []
    for row in choices:
        year = int(row["outer_validation_year"])
        first_inner = row["inner"][c1.VARIANTS[0]]["selection"]
        outer = row["outer_selection"]
        for phase, selection in (("inner", first_inner), ("outer", outer)):
            start = str(selection["train_start"])
            end = str(selection["train_end"])
            if pd.Timestamp(end) >= pd.Timestamp("2026-01-01"):
                raise RuntimeError("C9 diagnostic window crossed into 2026")
            summary = summarize_neutralization_variants(
                observations,
                start=start,
                end=end,
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
                    "train_start": start,
                    "train_end": end,
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

    payload = {
        "method": "fold_safe_neutralization_diagnostics_only",
        "selection_changed": False,
        "candidate_changed": False,
        "holdout_unlocked": False,
        "pre_2026_only": True,
        "canonical_c1_contracts": True,
        "core_factors_only": list(c1.CORE_ALPHA_FACTORS),
        "variants": list(c1.VARIANTS),
        "captured_variant_count": len(_CAPTURED),
        "fold_count": len(choices),
        "windows": windows,
    }
    (output / "c9_diagnostics_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> int:
    _assert_pre_2026_only(sys.argv[1:])
    _CAPTURED.clear()
    install_v5_c_contracts(c1)
    c1._variant_observations = _capture_variant_observations
    rc = c1.main()
    output = Path(_arg_value(sys.argv[1:], "--output") or "output/v5_c_nested")
    _build_fold_safe_diagnostics(output)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
