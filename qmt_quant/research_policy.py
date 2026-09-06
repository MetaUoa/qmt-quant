from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchDataPolicy:
    """Frozen fail-closed lower bounds for historical V5 research data quality."""

    min_symbol_coverage: float = 0.98
    min_session_coverage: float = 0.97
    min_exposure_coverage: float = 0.95
    min_symbols_per_date: int = 50


DEFAULT_RESEARCH_DATA_POLICY = ResearchDataPolicy()


def cli_value(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        raise RuntimeError(f"missing value for {name}")
    return argv[index + 1]


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
    if value < float(minimum):
        raise RuntimeError(
            f"{name}={value:g} is below frozen research minimum {float(minimum):g}"
        )
    return value


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
