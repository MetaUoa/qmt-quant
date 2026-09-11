from __future__ import annotations

import sys

import run_v5_c_nested_research as c1

from qmt_quant.research_policy import assert_pre_holdout_end
from qmt_quant.research_runtime import install_v5_c_contracts


_OriginalCoreAlphaPolicy = c1.CoreAlphaPolicy
_OriginalFrozenCandidate = c1.FrozenCandidate


def _assert_pre_2026_only(argv: list[str]) -> None:
    assert_pre_holdout_end(argv, context="C7 research")


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
    install_v5_c_contracts(c1)
    c1.CoreAlphaPolicy = _stability_policy
    c1.FrozenCandidate = _c7_frozen_candidate
    return c1.main()


if __name__ == "__main__":
    raise SystemExit(main())
