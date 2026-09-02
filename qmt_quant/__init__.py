"""QMT Quant Research Suite V3-V7: PIT backtest, research, robustness and guarded MiniQMT execution."""

from .backtest import BacktestResult, calculate_metrics, rebalance_schedule, run_backtest
from .config import AcceptanceConfig, CostConfig, DataConfig, StrategyConfig
from .reference_data import ReferenceData

__version__ = "3.7.0"

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
