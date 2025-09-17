from MarketMind import logger
from MarketMind.components.data_ingestion import DataIngestion
from MarketMind.config.configuration import ConfigurationManager

STAGE_NAME = "DATA_INGESTION"


class DataIngestionPipeline:
    """
    Data Ingestion Pipeline to run the data ingestion process.
    """

    def __init__(self):
        self.config = ConfigurationManager().get_data_ingestion_config()
        self.data_ingestion = DataIngestion(config=self.config)

    def run(self):
        data = self.data_ingestion.run()
        return data


if __name__ == "__main__":
    try:
        pipeline = DataIngestionPipeline()
        logger.info(f"\n\n>>>>> Stage {STAGE_NAME} started <<<<<\n\n")
        data = pipeline.run()
        logger.info(f"\n\n>>>>> Stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
