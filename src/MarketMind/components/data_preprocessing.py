import numpy as np
import pandas as pd

from MarketMind import logger
from MarketMind.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    """
    This class handles data preprocessing for financial time series data.
    """

    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def run(self):
        """
        Executes the data preprocessing steps:
        - Loads raw data from CSV.
        - Handles missing values.
        - Creates technical indicators and features.
        - Saves the preprocessed data to a new CSV file.
        """
        logger.info("Starting data preprocessing...")
        df = pd.read_csv(self.config.data_path, index_col=0, parse_dates=True)
        logger.info("Handling missing values...")
        df.asfreq("B")
        df = df.ffill()
        logger.info("Creating technical indicators and features...")
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["SMA_short"] = df["Close"].rolling(window=20).mean()
        df["SMA_long"] = df["Close"].rolling(window=50).mean()
        df["volatility"] = df["log_return"].rolling(window=20).std()
        df["open_close_ratio"] = df["Open"] / df["Close"]
        df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["volume_zscore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df[
            "Volume"
        ].rolling(20).std()
        df.dropna(inplace=True)

        train_size = int(len(df) * self.config.train_size)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        train_df.to_csv(self.config.preprocessed_train_data_path, index=True)
        test_df.to_csv(self.config.preprocessed_test_data_path, index=True)

        logger.info("Data preprocessing completed and files saved.")
        return df
