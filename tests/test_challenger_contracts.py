import pytest

from qmt_quant.challenger_contracts import (
    ChallengerCandidate,
    predeclared_challenger_candidates,
    validate_challenger_candidates,
)
from qmt_quant.core_alpha import CHALLENGER_FACTORS, CORE_ALPHA_FACTORS


def test_c8_candidates_are_core_plus_at_most_one_challenger():
    candidates = predeclared_challenger_candidates()
    assert candidates[0].name == "core"
    assert candidates[0].factors == CORE_ALPHA_FACTORS
    assert len(candidates) == 1 + len(CHALLENGER_FACTORS)
    for row in candidates[1:]:
        added = set(row.factors).difference(CORE_ALPHA_FACTORS)
        assert len(added) == 1
        assert added.issubset(CHALLENGER_FACTORS)


def test_c8_contract_rejects_failed_momentum_reintroduction():
    bad = (
        ChallengerCandidate(
            "bad",
            CORE_ALPHA_FACTORS + ("momentum_20_5",),
        ),
    )
    with pytest.raises(ValueError, match="excluded factors"):
        validate_challenger_candidates(bad)
