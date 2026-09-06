from __future__ import annotations

import math

import pytest

from qmt_quant.research_policy import (
    DEFAULT_RESEARCH_DATA_POLICY,
    assert_cli_float_floor,
    assert_cli_int_floor,
    assert_data_audit_thresholds,
    assert_float_floor_value,
)
from run_v5_b_canonical_research import _assert_data_policy as assert_b_policy
from run_v5_c_canonical_research import _assert_data_policy as assert_c_policy
from run_v5_c9_neutralization_diagnostics import _assert_data_policy as assert_c9_policy
from run_v5_composite_canonical_oos import _assert_data_policy as assert_composite_policy


def test_default_research_data_policy_is_frozen() -> None:
    policy = DEFAULT_RESEARCH_DATA_POLICY
    assert policy.min_symbol_coverage == 0.98
    assert policy.min_session_coverage == 0.97
    assert policy.min_exposure_coverage == 0.95
    assert policy.min_symbols_per_date == 50


def test_cli_floor_helpers_allow_equal_or_stricter_values() -> None:
    assert assert_cli_float_floor(
        ["--min-symbol-coverage", "0.99"],
        "--min-symbol-coverage",
        minimum=0.98,
        default=0.98,
    ) == 0.99
    assert assert_cli_int_floor(
        ["--min-symbols-per-date", "60"],
        "--min-symbols-per-date",
        minimum=50,
        default=50,
    ) == 60


def test_cli_floor_helpers_reject_loosened_thresholds() -> None:
    with pytest.raises(RuntimeError, match="below frozen research minimum"):
        assert_cli_float_floor(
            ["--min-symbol-coverage", "0.97"],
            "--min-symbol-coverage",
            minimum=0.98,
            default=0.98,
        )
    with pytest.raises(RuntimeError, match="below frozen research minimum"):
        assert_cli_int_floor(
            ["--min-symbols-per-date", "49"],
            "--min-symbols-per-date",
            minimum=50,
            default=50,
        )


def test_float_floor_rejects_non_finite_values() -> None:
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(RuntimeError, match="must be finite"):
            assert_float_floor_value(value, "--min-symbol-coverage", minimum=0.98)
        with pytest.raises(RuntimeError, match="must be finite"):
            assert_cli_float_floor(
                ["--min-symbol-coverage", str(value)],
                "--min-symbol-coverage",
                minimum=0.98,
                default=0.98,
            )


def test_direct_data_audit_thresholds_share_frozen_policy() -> None:
    assert assert_data_audit_thresholds(
        min_symbol_coverage=0.99,
        min_session_coverage=0.98,
    ) == (0.99, 0.98)
    with pytest.raises(RuntimeError, match="min-symbol-coverage"):
        assert_data_audit_thresholds(
            min_symbol_coverage=0.97,
            min_session_coverage=0.97,
        )
    with pytest.raises(RuntimeError, match="min-session-coverage"):
        assert_data_audit_thresholds(
            min_symbol_coverage=0.98,
            min_session_coverage=0.96,
        )


def test_canonical_runners_refuse_relaxed_data_floors() -> None:
    with pytest.raises(RuntimeError):
        assert_b_policy(["--min-symbol-coverage", "0.97"])
    with pytest.raises(RuntimeError):
        assert_composite_policy(["--min-symbol-coverage", "0.97"])
    with pytest.raises(RuntimeError):
        assert_c_policy(["--min-exposure-coverage", "0.94"])
    with pytest.raises(RuntimeError):
        assert_c9_policy(["--min-symbols-per-date", "49"])


def test_canonical_runners_accept_frozen_defaults_without_cli_overrides() -> None:
    assert_b_policy([])
    assert_composite_policy([])
    assert_c_policy([])
    assert_c9_policy([])
