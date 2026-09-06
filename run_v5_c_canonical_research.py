from __future__ import annotations

import sys

import run_v5_c_nested_research as c1

from qmt_quant.research_policy import (
    DEFAULT_RESEARCH_DATA_POLICY,
    assert_cli_float_floor,
    assert_cli_int_floor,
)
from qmt_quant.research_runtime import install_v5_c_contracts


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


def main() -> int:
    _assert_data_policy(sys.argv[1:])
    install_v5_c_contracts(c1)
    return c1.main()


if __name__ == "__main__":
    raise SystemExit(main())
