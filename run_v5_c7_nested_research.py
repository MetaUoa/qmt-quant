from __future__ import annotations

import sys

import run_v5_c_nested_research as c1


_MAX_RESEARCH_END = "20251231"
_OriginalCoreAlphaPolicy = c1.CoreAlphaPolicy
_OriginalFrozenCandidate = c1.FrozenCandidate


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
        raise RuntimeError("C7 research end must be YYYYMMDD or YYYY-MM-DD")
    if end > _MAX_RESEARCH_END:
        raise RuntimeError("C7 is pre-2026 research only; holdout remains blinded")


def _stability_policy(*args, **kwargs):
    kwargs["include_challengers"] = False
    kwargs["stability_weighting"] = True
    return _OriginalCoreAlphaPolicy(*args, **kwargs)


def _c7_frozen_candidate(*args, **kwargs):
    kwargs["name"] = "v5-c7-core-stability-neutralized"
    kwargs["research_data_end"] = "2025-12-31"
    return _OriginalFrozenCandidate(*args, **kwargs)


def main() -> int:
    _assert_pre_2026_only(sys.argv[1:])
    c1.CoreAlphaPolicy = _stability_policy
    c1.FrozenCandidate = _c7_frozen_candidate
    return c1.main()


if __name__ == "__main__":
    raise SystemExit(main())
