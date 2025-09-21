from dataclasses import dataclass
from pathlib import Path

# config entities for each stage in pipeline


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_path: Path
    asset: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    data_path: Path
    preprocessed_data_dir: Path
    preprocessed_train_data_path: Path
    preprocessed_test_data_path: Path
    train_size: float
    sma_short_window: int
    sma_long_window: int
    volatility_window: int
    volume_window: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    preprocessed_train_data_path: Path
    model_dir: Path
    tb_log_dir: Path
    window_size: 100
    transaction_cost: float
    initial_balance: float
    reward_scaling: float
    turnover_penalty: float
    net_arch: list
    initial_lr: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    ent_coef: float
    eval_freq: int
    total_timesteps: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    preprocessed_test_data_path: Path
    model_dir: Path
    normalized_vec_env_path: Path
    report_dir: Path
    plots_dir: Path
    window_size: int
    transaction_cost: float
    initial_balance: float
    reward_scaling: float
    turnover_penalty: float
