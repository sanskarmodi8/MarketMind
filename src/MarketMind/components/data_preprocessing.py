import os
from pathlib import Path

import numpy as np
import pandas as pd

from MarketMind import logger
from MarketMind.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    """
    Handles data preprocessing for financial time series data.
    Includes:
        - Loading raw data
        - Missing value handling
        - Feature/indicator creation
        - Train/test split & saving
    """

    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features/indicators on the dataframe.
        """
        logger.info("Creating technical indicators and features...")

        # Basic return & volatility features
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        df["SMA_short"] = (
            df["Close"].rolling(window=self.config.sma_short_window).mean()
        )
        df["SMA_long"] = df["Close"].rolling(window=self.config.sma_long_window).mean()

        df["volatility"] = (
            df["log_return"].rolling(window=self.config.volatility_window).std()
        )

        # Ratios and ranges
        df["open_close_ratio"] = df["Open"] / df["Close"]
        df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]

        # Volume z-score
        vol_mean = df["Volume"].rolling(self.config.volume_window).mean()
        vol_std = df["Volume"].rolling(self.config.volume_window).std()
        df["volume_zscore"] = (df["Volume"] - vol_mean) / (vol_std)

        return df

    def run(self) -> pd.DataFrame:
        """
        Executes preprocessing steps:
        - Load raw data from CSV.
        - Handle missing values.
        - Create technical indicators/features.
        - Split train/test and save to CSV.
        """
        logger.info("Starting data preprocessing...")

        # Load raw data
        df = pd.read_csv(self.config.data_path, index_col=0, parse_dates=True)
        logger.info(
            f"Loaded raw data from {self.config.data_path} with shape {df.shape}"
        )

        # Handle missing values and set business day frequency
        logger.info("Handling missing values and reindexing to business days...")
        df = df.asfreq("B")
        df = df.ffill()

        # Create features
        df = self._create_features(df)

        # Drop rows with NaN after feature engineering
        df.dropna(inplace=True)
        logger.info(f"Data shape after feature creation and dropna: {df.shape}")

        # Train/test split
        train_size = int(len(df) * self.config.train_size)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        # Ensure directories exist
        os.makedirs(
            Path(self.config.preprocessed_train_data_path).parent, exist_ok=True
        )
        os.makedirs(Path(self.config.preprocessed_test_data_path).parent, exist_ok=True)

        # Save processed datasets
        train_df.to_csv(self.config.preprocessed_train_data_path, index=True)
        test_df.to_csv(self.config.preprocessed_test_data_path, index=True)

        logger.info(
            f"Data preprocessing completed. "
            f"Train data saved to {self.config.preprocessed_train_data_path} "
            f"({train_df.shape}), Test data saved to {self.config.preprocessed_test_data_path} "
            f"({test_df.shape})"
        )

        return df
