from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qmt_quant.neutralization import (
    exposure_correlations,
    neutralize_cross_section,
    neutralize_panel,
)


def test_neutralize_removes_group_and_continuous_exposure():
    index = [f"s{i:03d}" for i in range(100)]
    size = pd.Series(np.linspace(-2.0, 2.0, 100), index=index)
    groups = pd.Series(["A"] * 50 + ["B"] * 50, index=index)
    group_effect = groups.map({"A": -1.0, "B": 1.0}).astype(float)
    alpha = pd.Series(np.sin(np.linspace(0.0, 6.0, 100)), index=index)
    factor = 2.5 * size + 3.0 * group_effect + alpha

    residual = neutralize_cross_section(
        factor,
        groups=groups,
        exposures=pd.DataFrame({"size": size}),
        min_symbols=50,
    )
    correlations = exposure_correlations(residual, pd.DataFrame({"size": size}))
    assert abs(correlations["size"]) < 1e-10
    assert abs(residual.loc[groups.eq("A")].mean()) < 1e-10
    assert abs(residual.loc[groups.eq("B")].mean()) < 1e-10


def test_neutralize_fails_closed_on_missing_exposure_coverage():
    index = [f"s{i:03d}" for i in range(100)]
    factor = pd.Series(np.arange(100, dtype=float), index=index)
    exposure = pd.Series(np.arange(100, dtype=float), index=index)
    exposure.iloc[:20] = np.nan
    with pytest.raises(RuntimeError, match="coverage"):
        neutralize_cross_section(
            factor,
            exposures=pd.DataFrame({"size": exposure}),
            min_symbols=50,
            min_coverage=0.95,
        )


def test_neutralize_fails_closed_on_missing_group_coverage():
    index = [f"s{i:03d}" for i in range(100)]
    factor = pd.Series(np.arange(100, dtype=float), index=index)
    groups = pd.Series(["A"] * 50 + ["B"] * 50, index=index, dtype="string")
    groups.iloc[:10] = pd.NA
    with pytest.raises(RuntimeError, match="coverage"):
        neutralize_cross_section(
            factor,
            groups=groups,
            min_symbols=50,
            min_coverage=0.95,
        )


def test_panel_requires_each_exposure_snapshot():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    columns = [f"s{i:03d}" for i in range(40)]
    factor = pd.DataFrame(np.random.default_rng(1).normal(size=(2, 40)), index=dates, columns=columns)
    exposure = pd.DataFrame(
        np.random.default_rng(2).normal(size=(1, 40)), index=dates[:1], columns=columns
    )
    with pytest.raises(RuntimeError, match="missing exposure snapshot"):
        neutralize_panel(
            factor,
            exposure_panels={"size": exposure},
            min_symbols=30,
            min_coverage=0.95,
        )
