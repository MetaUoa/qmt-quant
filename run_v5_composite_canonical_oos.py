from __future__ import annotations

import sys

import run_v5_composite_oos as legacy
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
        raise RuntimeError("V5 composite research end must be YYYYMMDD or YYYY-MM-DD")
    if end > _MAX_RESEARCH_END:
        raise RuntimeError("V5 composite is pre-2026 research only; holdout remains blinded")


def main() -> int:
    _assert_pre_2026_only(sys.argv[1:])
    install_legacy_v5_research_contracts(legacy, context="V5 score")
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
