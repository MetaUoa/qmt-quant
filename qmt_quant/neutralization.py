from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _validate_complete(
    y: pd.Series,
    complete: pd.Series,
    *,
    min_symbols: int,
    min_coverage: float,
) -> None:
    denominator = int(y.notna().sum())
    coverage = float(complete.sum() / denominator) if denominator else 0.0
    if coverage < float(min_coverage):
        raise RuntimeError(
            f"neutralization exposure coverage {coverage:.2%} is below required {min_coverage:.2%}"
        )
    if int(complete.sum()) < int(min_symbols):
        raise RuntimeError(
            f"only {int(complete.sum())} complete neutralization rows; require {min_symbols}"
        )


def _standardize_exposures(
    exposures: pd.DataFrame,
    index: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    numeric = exposures.reindex(index).apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1)
    standardized = numeric.copy()
    for column in standardized.columns:
        series = standardized[column]
        std = float(series.std(ddof=0))
        mean = float(series.mean()) if series.notna().any() else float("nan")
        if np.isfinite(std) and std > 0.0 and np.isfinite(mean):
            standardized[column] = (series - mean) / std
        else:
            standardized[column] = np.nan
            complete &= False
    return standardized, complete


def neutralize_cross_section(
    factor: pd.Series,
    *,
    groups: pd.Series | None = None,
    exposures: pd.DataFrame | None = None,
    min_symbols: int = 30,
    min_coverage: float = 0.95,
) -> pd.Series:
    """Residualize one date's factor against supplied cross-sectional exposures.

    The caller must provide exposures that were known on the same date. Missing
    exposure coverage is fail-closed. Industry/category fixed effects are removed
    by within-group demeaning (Frisch-Waugh-Lovell) rather than constructing a wide
    dummy matrix; this is algebraically equivalent to OLS fixed effects and keeps
    full-market research tractable.
    """
    y = pd.to_numeric(factor, errors="coerce").rename("factor")
    valid_factor = y.dropna()
    if len(valid_factor) < int(min_symbols):
        raise RuntimeError(f"only {len(valid_factor)} factor observations; require {min_symbols}")

    group = None
    complete = y.notna().copy()
    if groups is not None:
        group = groups.reindex(y.index).astype("string")
        complete &= group.notna()

    standardized = None
    if exposures is not None:
        standardized, exposure_complete = _standardize_exposures(exposures, y.index)
        complete &= exposure_complete

    if groups is None and exposures is None:
        centered = valid_factor - float(valid_factor.mean())
        return centered.reindex(y.index)

    _validate_complete(
        y,
        complete,
        min_symbols=min_symbols,
        min_coverage=min_coverage,
    )
    target = y.loc[complete].astype(float)
    result = pd.Series(np.nan, index=y.index, dtype=float)

    if group is not None:
        group_complete = group.loc[complete]
        target_within = target - target.groupby(group_complete).transform("mean")
        if standardized is None or standardized.shape[1] == 0:
            result.loc[complete] = target_within.to_numpy()
            return result

        x = standardized.loc[complete].astype(float)
        x_within = x.copy()
        for column in x.columns:
            x_within[column] = x[column] - x[column].groupby(group_complete).transform("mean")
        matrix = x_within.to_numpy()
        beta, *_ = np.linalg.lstsq(matrix, target_within.to_numpy(), rcond=None)
        residual = target_within.to_numpy() - matrix @ beta
        result.loc[complete] = residual
        return result

    if standardized is None:
        raise RuntimeError("internal neutralization exposure state is missing")
    x = standardized.loc[complete].astype(float)
    x.insert(0, "intercept", 1.0)
    beta, *_ = np.linalg.lstsq(x.to_numpy(), target.to_numpy(), rcond=None)
    residual = target.to_numpy() - x.to_numpy() @ beta
    result.loc[complete] = residual
    return result


def neutralize_panel(
    factor: pd.DataFrame,
    *,
    group_panel: pd.DataFrame | None = None,
    exposure_panels: Mapping[str, pd.DataFrame] | None = None,
    min_symbols: int = 30,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Date-local neutralization with no information carried across timestamps."""
    output = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns, dtype=float)
    for ts in factor.index:
        groups = group_panel.loc[ts] if group_panel is not None and ts in group_panel.index else None
        exposures = None
        if exposure_panels:
            columns: dict[str, pd.Series] = {}
            for name, panel in exposure_panels.items():
                if ts not in panel.index:
                    raise RuntimeError(f"missing exposure snapshot {name} on {pd.Timestamp(ts).date()}")
                columns[str(name)] = panel.loc[ts]
            exposures = pd.DataFrame(columns)
        if group_panel is not None and groups is None:
            raise RuntimeError(f"missing group snapshot on {pd.Timestamp(ts).date()}")
        output.loc[ts] = neutralize_cross_section(
            factor.loc[ts],
            groups=groups,
            exposures=exposures,
            min_symbols=min_symbols,
            min_coverage=min_coverage,
        )
    return output


def exposure_correlations(
    residual: pd.Series,
    exposures: pd.DataFrame,
) -> dict[str, float]:
    """Diagnostic correlations after residualization."""
    result: dict[str, float] = {}
    for column in exposures.columns:
        pair = pd.concat(
            [
                pd.to_numeric(residual, errors="coerce"),
                pd.to_numeric(exposures[column], errors="coerce"),
            ],
            axis=1,
        ).dropna()
        value = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) > 1 else np.nan
        result[str(column)] = float(value) if pd.notna(value) else float("nan")
    return result
