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
    min_orientation_dates: int = 24
    require_same_sign_across_horizons: bool = True
    duplicate_value_columns: tuple[str, ...] = ("rank_ic", "top_bottom_spread")
    duplicate_atol: float = 1e-12
    max_abs_correlation: float = 0.80
    min_factors: int = 2
    max_factors: int = 4
    weight_metric_cap: float = 0.10


DEFAULT_V5_SELECTION_POLICY = V5SelectionPolicy()
MAX_PRE_HOLDOUT_END = "20251231"


def cli_values(argv: list[str], name: str) -> list[str]:
    """Return all values supplied for one long CLI option.

    Both ``--name value`` and ``--name=value`` are accepted. Multiple occurrences
    are retained so policy checks can reject argparse's otherwise silent last-value
    wins behavior.
    """
    values: list[str] = []
    prefix = f"{name}="
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv):
                raise RuntimeError(f"missing value for {name}")
            values.append(argv[index + 1])
            index += 2
            continue
        if token.startswith(prefix):
            raw = token[len(prefix) :]
            if raw == "":
                raise RuntimeError(f"missing value for {name}")
            values.append(raw)
        index += 1
    return values


def cli_value(argv: list[str], name: str) -> str | None:
    values = cli_values(argv, name)
    if len(values) > 1:
        raise RuntimeError(f"duplicate {name} arguments are not allowed")
    return values[0] if values else None


def normalize_research_end(value: str, *, name: str = "--end") -> str:
    normalized = str(value).strip().replace("-", "")
    if not normalized.isdigit() or len(normalized) != 8:
        raise RuntimeError(f"{name} must be YYYYMMDD or YYYY-MM-DD")
    return normalized


def assert_pre_holdout_end(
    argv: list[str],
    *,
    name: str = "--end",
    default: str = MAX_PRE_HOLDOUT_END,
    context: str = "V5 research",
) -> str:
    """Fail closed if the effective research end can cross into the 2026 holdout."""
    raw = cli_value(argv, name)
    normalized = normalize_research_end(default if raw is None else raw, name=name)
    if normalized > MAX_PRE_HOLDOUT_END:
        raise RuntimeError(f"{context} is pre-2026 research only; holdout remains blinded")
    return normalized


def assert_pre_holdout_end_value(value: str, *, context: str = "V5 research") -> str:
    normalized = normalize_research_end(value)
    if normalized > MAX_PRE_HOLDOUT_END:
        raise RuntimeError(f"{context} is pre-2026 research only; holdout remains blinded")
    return normalized


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
