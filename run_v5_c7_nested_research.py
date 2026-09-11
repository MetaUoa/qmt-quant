from __future__ import annotations

import sys

import run_v5_c_nested_research as c1

from qmt_quant.research_policy import (
    DEFAULT_RESEARCH_DATA_POLICY,
    assert_cli_float_floor,
    assert_cli_int_floor,
    assert_pre_holdout_end,
)
from qmt_quant.research_runtime import install_v5_c_contracts


_OriginalCoreAlphaPolicy = c1.CoreAlphaPolicy
_OriginalFrozenCandidate = c1.FrozenCandidate


def _assert_pre_2026_only(argv: list[str]) -> None:
    assert_pre_holdout_end(argv, context="C7 research")


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


def _stability_policy(*args, **kwargs):
    kwargs["include_challengers"] = False
    kwargs["stability_weighting"] = True
    return _OriginalCoreAlphaPolicy(*args, **kwargs)


def _c7_frozen_candidate(*args, **kwargs):
    kwargs["name"] = "v5-c7-core-stability-neutralized"
    kwargs["research_data_end"] = "2025-12-31"
    return _OriginalFrozenCandidate(*args, **kwargs)


def main() -> int:
    argv = sys.argv[1:]
    _assert_pre_2026_only(argv)
    _assert_data_policy(argv)
    install_v5_c_contracts(c1)
    c1.CoreAlphaPolicy = _stability_policy
    c1.FrozenCandidate = _c7_frozen_candidate
    return c1.main()


if __name__ == "__main__":
    raise SystemExit(main())
