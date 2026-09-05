from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .composites import CompositeSpec, apply_composite
from .neutralization import neutralize_panel


@dataclass(frozen=True)
class NeutralizationInputs:
    industry_panel: pd.DataFrame | None = None
    size_panel: pd.DataFrame | None = None
    liquidity_panel: pd.DataFrame | None = None

    def available_variants(self) -> tuple[str, ...]:
        variants = ["raw"]
        if self.liquidity_panel is not None:
            variants.append("liquidity")
        if self.industry_panel is not None:
            variants.append("industry")
        if self.industry_panel is not None and self.size_panel is not None and self.liquidity_panel is not None:
            variants.append("industry_size_liquidity")
        return tuple(variants)


def _variant_exposures(
    variant: str,
    inputs: NeutralizationInputs,
) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame]]:
    if variant == "raw":
        return None, {}
    if variant == "liquidity":
        if inputs.liquidity_panel is None:
            raise RuntimeError("liquidity neutralization requested without PIT liquidity exposure")
        return None, {"liquidity": inputs.liquidity_panel}
    if variant == "industry":
        if inputs.industry_panel is None:
            raise RuntimeError("industry neutralization requested without PIT industry snapshots")
        return inputs.industry_panel, {}
    if variant == "industry_size_liquidity":
        if inputs.industry_panel is None or inputs.size_panel is None or inputs.liquidity_panel is None:
            raise RuntimeError(
                "full neutralization requires PIT industry, size and liquidity exposures"
            )
        return inputs.industry_panel, {
            "size": inputs.size_panel,
            "liquidity": inputs.liquidity_panel,
        }
    raise ValueError(f"unknown neutralization variant: {variant}")


def neutralize_factor_panels(
    factor_panels: Mapping[str, pd.DataFrame],
    *,
    variant: str,
    inputs: NeutralizationInputs,
    min_symbols: int = 50,
    min_coverage: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Residualize every supplied factor using only same-date PIT exposures."""
    if variant == "raw":
        return {str(name): panel.copy() for name, panel in factor_panels.items()}
    groups, exposures = _variant_exposures(variant, inputs)
    output: dict[str, pd.DataFrame] = {}
    for name, panel in factor_panels.items():
        output[str(name)] = neutralize_panel(
            panel,
            group_panel=groups,
            exposure_panels=exposures or None,
            min_symbols=min_symbols,
            min_coverage=min_coverage,
        )
    return output


def build_neutralized_composite(
    factor_panels: Mapping[str, pd.DataFrame],
    spec: CompositeSpec,
    *,
    variant: str,
    inputs: NeutralizationInputs,
    min_symbols: int = 50,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    neutralized = neutralize_factor_panels(
        factor_panels,
        variant=variant,
        inputs=inputs,
        min_symbols=min_symbols,
        min_coverage=min_coverage,
    )
    return apply_composite(neutralized, spec)
