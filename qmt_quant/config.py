from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    start: str = "20180101"
    end: str = "20251231"
    sector: str = "沪深A股"
    benchmark: str = "000905.SH"
    dividend_type: str = "front"
    batch_size: int = 200
    reference_dir: str = "data/reference"
    bar_cache_dir: str = "data/qmt_bars"


@dataclass(frozen=True)
class StrategyConfig:
    mom_short: int = 20
    mom_mid: int = 60
    mom_long: int = 120
    ma_fast: int = 20
    ma_slow: int = 60
    vol_window: int = 20
    amount_window: int = 20
    benchmark_ma: int = 120
    benchmark_mom_days: int = 20
    breadth_ma: int = 60
    top_n: int = 8
    rebalance_days: int = 5
    execution_delay_sessions: int = 1
    min_price: float = 3.0
    min_amount: float = 20_000_000.0
    min_momentum: float = 0.02
    max_daily_vol: float = 0.075
    min_breadth: float = 0.0
    benchmark_mom_floor: float = -0.03
    risk_off_exposure: float = 0.0
    min_listing_sessions: int = 120
    weight_short: float = 0.20
    weight_mid: float = 0.30
    weight_long: float = 0.50
    vol_penalty: float = 0.75

    @property
    def warmup(self) -> int:
        return max(
            self.mom_long,
            self.ma_slow,
            self.vol_window,
            self.amount_window,
            self.benchmark_ma,
            self.breadth_ma,
            self.min_listing_sessions,
        ) + 5


@dataclass(frozen=True)
class CostConfig:
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    slippage_bps: float = 5.0
    lot_size: int = 100
    limit_tolerance: float = 0.001
    fill_probability: float = 1.0
    fill_seed: int = 20260902


@dataclass(frozen=True)
class AcceptanceConfig:
    target_multiple: float = 150.0
    grade_b_multiple: float = 50.0
    max_drawdown_a: float = 0.35
    max_drawdown_b: float = 0.45
    min_sharpe_a: float = 1.50
    min_sharpe_b: float = 1.00
    min_oos_cagr_a: float = 0.30
    min_oos_cagr_b: float = 0.15
    min_positive_oos_folds_a: int = 4
    min_positive_oos_folds_b: int = 3
    min_stress_pass_ratio_a: float = 0.75
    min_stress_pass_ratio_b: float = 0.60
