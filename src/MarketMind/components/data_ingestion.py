import yfinance as yf

from MarketMind import logger
from MarketMind.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """
    Class to handle data ingestion from Yahoo Finance.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def run(self):
        """
        Downloads historical market data for the specified asset and date range,
        and saves it to the specified data path.
        """
        logger.info(
            f"Starting data download for {self.config.asset} from {self.config.start_date} to {self.config.end_date}"
        )
        data = yf.download(
            self.config.asset, start=self.config.start_date, end=self.config.end_date
        )
        if data.empty:
            logger.warning(
                f"No data found for {self.config.asset} in the given date range."
            )
        else:
            data.columns = data.columns.droplevel(
                1
            )  # Drop multi-level column if exists
            data.to_csv(self.config.data_path, index=True)
            logger.info(f"Data downloaded and saved to {self.config.data_path}")
        return data
