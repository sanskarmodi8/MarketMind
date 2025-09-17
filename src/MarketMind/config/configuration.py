from MarketMind.constants import *
from MarketMind.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    ModelEvaluationConfig,
    ModelTrainingConfig,
)
from MarketMind.utils.common import create_directories, read_yaml

# final configuration for all stages in the pipeline


class ConfigurationManager:
    def __init__(self):
        # load configs and params, and create root dir for artifacts
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARMS_FILE_PATH)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        # prepare config for data ingestion
        config = self.config.data_ingestion
        params = self.params.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            asset=params.asset,
            start_date=params.start_date,
            end_date=params.end_date,
        )

    def get_data_preprocessing_config(self):
        # prepare config for data preprocessing
        config = self.config.data_preprocessing
        params = self.params.data_preprocessing
        create_directories([config.root_dir, config.preprocessed_data_dir])
        return DataPreprocessingConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            preprocessed_data_dir=config.preprocessed_data_dir,
            preprocessed_train_data_path=config.preprocessed_train_data_path,
            preprocessed_test_data_path=config.preprocessed_test_data_path,
            train_size=params.train_size,
        )

    def get_model_training_config(self):
        # prepare config for model training
        config = self.config.model_training
        params = self.params.model_training
        create_directories([config.root_dir, config.tb_log_dir, config.model_dir])
        return ModelTrainingConfig(
            root_dir=config.root_dir,
            preprocessed_train_data_path=config.preprocessed_train_data_path,
            model_dir=config.model_dir,
            tb_log_dir=config.tb_log_dir,
            normalized_vec_env_path=config.normalized_vec_env_path,
            window_size=params.window_size,
            transaction_cost=params.transaction_cost,
            initial_balance=params.initial_balance,
            reward_scaling=params.reward_scaling,
            turnover_penalty=params.turnover_penalty,
            net_arch=params.net_arch,
            initial_lr=params.initial_lr,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            n_epochs=params.n_epochs,
            gamma=params.gamma,
            ent_coef=params.ent_coef,
            eval_freq=params.eval_freq,
            total_timesteps=params.total_timesteps,
        )

    def get_model_evaluation_config(self):
        # prepare config for model evaluation
        config = self.config.model_evaluation
        params = self.params.model_training
        create_directories([config.root_dir, config.report_dir, config.plots_dir])
        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            preprocessed_test_data_path=config.preprocessed_test_data_path,
            model_dir=config.model_dir,
            normalized_vec_env_path=config.normalized_vec_env_path,
            report_dir=config.report_dir,
            plots_dir=config.plots_dir,
            window_size=params.window_size,
            transaction_cost=params.transaction_cost,
            initial_balance=params.initial_balance,
            reward_scaling=params.reward_scaling,
            turnover_penalty=params.turnover_penalty,
        )
