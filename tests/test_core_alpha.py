from qmt_quant.core_alpha import (
    CHALLENGER_FACTORS,
    CORE_ALPHA_FACTORS,
    EXCLUDED_CORE_FACTORS,
    CoreAlphaPolicy,
)


def test_core_alpha_default_pool_is_small_and_excludes_failed_momentum():
    policy = CoreAlphaPolicy()
    assert policy.allowed_factors == CORE_ALPHA_FACTORS
    assert set(CORE_ALPHA_FACTORS).isdisjoint(EXCLUDED_CORE_FACTORS)
    assert "momentum_20_5" not in policy.allowed_factors
    assert "liquidity_stability" in policy.allowed_factors
    assert "low_volatility" in policy.allowed_factors


def test_challengers_are_opt_in_only():
    default = CoreAlphaPolicy()
    challenger = CoreAlphaPolicy(include_challengers=True)
    assert set(CHALLENGER_FACTORS).isdisjoint(default.allowed_factors)
    assert challenger.allowed_factors == CORE_ALPHA_FACTORS + CHALLENGER_FACTORS
