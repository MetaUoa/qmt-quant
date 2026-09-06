from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ResearchDataPolicy:
    """Frozen fail-closed lower bounds for historical V5 research data quality."""

    min_symbol_coverage: float = 0.98
    min_session_coverage: float = 0.97
    min_exposure_coverage: float = 0.95
    min_symbols_per_date: int = 50


DEFAULT_RESEARCH_DATA_POLICY = ResearchDataPolicy()


@dataclass(frozen=True)
class V5SelectionPolicy:
    """Frozen defaults for the historical training-only V5 composite selector.

    These are centralized invariants, not a request to retune factor selection. The
    selector keeps accepting explicit overrides for controlled historical experiments,
    but its default behavior is defined in one typed policy object.
    """

    safe_factors: tuple[str, ...] = (
        "low_volatility",
        "low_downside_risk",
        "liquidity_stability",
        "short_reversal",
        "momentum_20_5",
        "momentum_60_5",
        "momentum_120_5",
        "trend_quality",
        "trend_persistence",
    )
    correlation_horizon: int = 20
    min_abs_rank_ic: float = 0.01
    max_abs_correlation: float = 0.80
    min_factors: int = 2
    max_factors: int = 4
    weight_metric_cap: float = 0.10


DEFAULT_V5_SELECTION_POLICY = V5SelectionPolicy()


def cli_value(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        raise RuntimeError(f"missing value for {name}")
    return argv[index + 1]


def assert_float_floor_value(value: float, name: str, *, minimum: float) -> float:
    """Reject non-finite or loosened research thresholds."""
    numeric = float(value)
    floor = float(minimum)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{name} must be finite, got {numeric!r}")
    if numeric < floor:
        raise RuntimeError(f"{name}={numeric:g} is below frozen research minimum {floor:g}")
    return numeric


def assert_cli_float_floor(
    argv: list[str],
    name: str,
    *,
    minimum: float,
    default: float,
) -> float:
    """Reject a canonical research invocation that loosens a frozen float threshold."""
    raw = cli_value(argv, name)
    try:
        value = float(default if raw is None else raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid numeric value for {name}: {raw}") from exc
    return assert_float_floor_value(value, name, minimum=minimum)


def assert_cli_int_floor(
    argv: list[str],
    name: str,
    *,
    minimum: int,
    default: int,
) -> int:
    """Reject a canonical research invocation that loosens a frozen integer threshold."""
    raw = cli_value(argv, name)
    try:
        value = int(default if raw is None else raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid integer value for {name}: {raw}") from exc
    if value < int(minimum):
        raise RuntimeError(
            f"{name}={value} is below frozen research minimum {int(minimum)}"
        )
    return value


def assert_data_audit_thresholds(
    *,
    min_symbol_coverage: float,
    min_session_coverage: float,
    policy: ResearchDataPolicy = DEFAULT_RESEARCH_DATA_POLICY,
) -> tuple[float, float]:
    """Validate direct historical data-audit thresholds against the frozen policy."""
    symbol = assert_float_floor_value(
        min_symbol_coverage,
        "--min-symbol-coverage",
        minimum=policy.min_symbol_coverage,
    )
    session = assert_float_floor_value(
        min_session_coverage,
        "--min-session-coverage",
        minimum=policy.min_session_coverage,
    )
    return symbol, session
