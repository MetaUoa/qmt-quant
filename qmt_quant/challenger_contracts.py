from __future__ import annotations

from dataclasses import dataclass

from .core_alpha import CHALLENGER_FACTORS, CORE_ALPHA_FACTORS, EXCLUDED_CORE_FACTORS


@dataclass(frozen=True)
class ChallengerCandidate:
    name: str
    factors: tuple[str, ...]


def predeclared_challenger_candidates() -> tuple[ChallengerCandidate, ...]:
    """Return the small, auditable C8 candidate set.

    The current C1 core remains the baseline. Each challenger is introduced one at a
    time so any improvement can be attributed without combinatorial data-mining.
    """
    candidates = [ChallengerCandidate("core", CORE_ALPHA_FACTORS)]
    for challenger in CHALLENGER_FACTORS:
        candidates.append(
            ChallengerCandidate(
                f"core_plus_{challenger}",
                CORE_ALPHA_FACTORS + (challenger,),
            )
        )
    validate_challenger_candidates(tuple(candidates))
    return tuple(candidates)


def validate_challenger_candidates(candidates: tuple[ChallengerCandidate, ...]) -> None:
    if not candidates:
        raise ValueError("at least one challenger candidate is required")
    names = [row.name for row in candidates]
    if len(names) != len(set(names)):
        raise ValueError("challenger candidate names must be unique")
    forbidden = set(EXCLUDED_CORE_FACTORS)
    allowed = set(CORE_ALPHA_FACTORS) | set(CHALLENGER_FACTORS)
    for row in candidates:
        factors = tuple(row.factors)
        if len(factors) != len(set(factors)):
            raise ValueError(f"duplicate factor in candidate {row.name}")
        unknown = set(factors).difference(allowed)
        if unknown:
            raise ValueError(f"candidate {row.name} contains unknown factors: {sorted(unknown)}")
        bad = set(factors).intersection(forbidden)
        if bad:
            raise ValueError(f"candidate {row.name} reintroduces excluded factors: {sorted(bad)}")
        if not set(CORE_ALPHA_FACTORS).issubset(factors):
            raise ValueError(f"candidate {row.name} must retain the complete C1 core")
        added = set(factors).difference(CORE_ALPHA_FACTORS)
        if len(added) > 1:
            raise ValueError(f"candidate {row.name} adds more than one challenger")
