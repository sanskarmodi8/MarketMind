from MarketMind import logger
from MarketMind.components.data_preprocessing import DataPreprocessing
from MarketMind.config.configuration import ConfigurationManager

STAGE_NAME = "DATA_PREPROCESSING"


class DataPreprocessingPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_data_preprocessing_config()
        self.data_preprocessing = DataPreprocessing(config=self.config)

    def run(self):
        df = self.data_preprocessing.run()
        return df


if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>> Stage {STAGE_NAME} started <<<<<\n\n")
        pipe = DataPreprocessingPipeline()
        data = pipe.run()
        logger.info(f"\n\n>>>>> Stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
