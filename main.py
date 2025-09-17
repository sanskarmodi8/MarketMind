from MarketMind import logger
from MarketMind.pipeline.stage_01_data_ingestion import DataIngestionPipeline

STAGE_NAME = "DATA_INGESTION"
try:
    pipe = DataIngestionPipeline()
    logger.info(f"\n\n>>>>> Stage {STAGE_NAME} started <<<<<\n\n")
    data = pipe.run()
    logger.info(f"\n\n>>>>> Stage {STAGE_NAME} completed <<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e