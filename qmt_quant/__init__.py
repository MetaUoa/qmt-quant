"""QMT Quant Research Suite: PIT backtest, free-data validation, research and guarded execution."""

from .backtest import BacktestResult, calculate_metrics, rebalance_schedule, run_backtest
from .config import AcceptanceConfig, CostConfig, DataConfig, StrategyConfig
from .reference_data import ReferenceData

__version__ = "4.0.0"

__all__ = [
    "AcceptanceConfig",
    "BacktestResult",
    "CostConfig",
    "DataConfig",
    "ReferenceData",
    "StrategyConfig",
    "calculate_metrics",
    "rebalance_schedule",
    "run_backtest",
]
