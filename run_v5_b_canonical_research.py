from __future__ import annotations

import sys

import run_v5_b_research as legacy
from qmt_quant.research_policy import (
    DEFAULT_RESEARCH_DATA_POLICY,
    assert_cli_float_floor,
    assert_cli_int_floor,
)
from qmt_quant.research_runtime import install_legacy_v5_research_contracts


_MAX_RESEARCH_END = "20251231"


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
        raise RuntimeError("V5-B research end must be YYYYMMDD or YYYY-MM-DD")
    if end > _MAX_RESEARCH_END:
        raise RuntimeError("V5-B is pre-2026 research only; holdout remains blinded")


def _assert_data_policy(argv: list[str]) -> None:
    policy = DEFAULT_RESEARCH_DATA_POLICY
    assert_cli_float_floor(
        argv,
        "--min-symbol-coverage",
        minimum=policy.min_symbol_coverage,
        default=policy.min_symbol_coverage,
    )
    assert_cli_int_floor(
        argv,
        "--min-symbols-per-date",
        minimum=policy.min_symbols_per_date,
        default=policy.min_symbols_per_date,
    )


def main() -> int:
    argv = sys.argv[1:]
    _assert_pre_2026_only(argv)
    _assert_data_policy(argv)
    install_legacy_v5_research_contracts(legacy, context="B research")
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
