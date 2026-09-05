import numpy as np
import pandas as pd

from qmt_quant.neutralization import neutralize_cross_section


def test_group_fixed_effect_neutralization_removes_group_and_continuous_exposure():
    n = 120
    groups = pd.Series(["A"] * 60 + ["B"] * 60)
    x = pd.Series(np.linspace(-2.0, 2.0, n))
    group_effect = np.where(groups.eq("A"), 5.0, -3.0)
    noise = np.sin(np.arange(n))
    factor = pd.Series(group_effect + 2.5 * x.to_numpy() + noise)
    residual = neutralize_cross_section(
        factor,
        groups=groups,
        exposures=pd.DataFrame({"size": x}),
        min_symbols=50,
        min_coverage=0.95,
    )
    clean = pd.DataFrame({"residual": residual, "group": groups, "x": x}).dropna()
    means = clean.groupby("group")["residual"].mean().abs()
    assert float(means.max()) < 1e-10
    assert abs(float(clean["residual"].corr(clean["x"]))) < 1e-10
