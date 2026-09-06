from __future__ import annotations

import pandas as pd
import pytest

from run_data_audit import count_observed_sessions, parse_args, validate_args


def test_count_observed_sessions_uses_same_denominator_sessions():
    frame = pd.DataFrame(
        {"close": [10.0, 10.1, 10.2]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    expected = pd.DatetimeIndex(["2025-01-02", "2025-01-06"])

    observed = count_observed_sessions(frame, expected, "close")

    assert observed == 2
    assert observed / len(expected) <= 1.0


def test_count_observed_sessions_ignores_nan_and_duplicate_rows():
    frame = pd.DataFrame(
        {"open": [10.0, float("nan"), 10.5, 10.6]},
        index=pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-06"]
        ),
    )
    expected = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"])

    assert count_observed_sessions(frame, expected, "open") == 2


def test_data_audit_defaults_use_frozen_research_policy() -> None:
    args = validate_args(parse_args([]))
    assert args.min_symbol_coverage == 0.98
    assert args.min_session_coverage == 0.97


def test_data_audit_rejects_relaxed_or_non_finite_thresholds() -> None:
    with pytest.raises(RuntimeError, match="min-symbol-coverage"):
        validate_args(parse_args(["--min-symbol-coverage", "0.97"]))
    with pytest.raises(RuntimeError, match="min-session-coverage"):
        validate_args(parse_args(["--min-session-coverage", "0.96"]))
    with pytest.raises(RuntimeError, match="must be finite"):
        validate_args(parse_args(["--min-symbol-coverage", "nan"]))
