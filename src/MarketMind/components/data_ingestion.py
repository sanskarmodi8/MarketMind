import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from MarketMind import logger
from MarketMind.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """
    Handles downloading historical market data from Yahoo Finance
    and saving it to the specified path.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        end_date = datetime.today().strftime("%Y-%m-%d") or self.config.end_date

    def run(self) -> pd.DataFrame:
        """
        Downloads historical data for the specified asset and date range,
        saves it to CSV, and returns the DataFrame.
        """
        logger.info(
            f"Starting data download for {self.config.asset} "
            f"from {self.config.start_date} to {self.config.end_date}"
        )

        try:
            data = yf.download(
                self.config.asset,
                start=self.config.start_date,
                end=self.config.end_date,
            )
        except Exception as e:
            logger.error(f"Error downloading data for {self.config.asset}: {e}")
            raise

        if data.empty:
            logger.warning(
                f"No data found for {self.config.asset} in the given date range "
                f"{self.config.start_date} to {self.config.end_date}."
            )
            return pd.DataFrame()

        # Handle MultiIndex columns
        data.columns = data.columns.droplevel(1)

        # Save to CSV
        data.to_csv(self.config.data_path, index=True)
        logger.info(f"Data downloaded and saved to {self.config.data_path}")

        return data
