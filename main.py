from MarketMind import logger
from MarketMind.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from MarketMind.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from MarketMind.pipeline.stage_03_model_training import ModelTrainingPipeline

# runs all the stages of the pipeline
# to run a specific stage, run the corresponding stage file

STAGE_NAME = "DATA_INGESTION"
try:
    pipe = DataIngestionPipeline()
    logger.info(f"\n\n>>>>> Stage {STAGE_NAME} started <<<<<\n\n")
    data = pipe.run()
    logger.info(f"\n\n>>>>> Stage {STAGE_NAME} completed <<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "DATA_PREPROCESSING"
try:
    logger.info(f"\n\n>>>>> Stage {STAGE_NAME} started <<<<<\n\n")
    pipe = DataPreprocessingPipeline()
    data = pipe.run()
    logger.info(f"\n\n>>>>> Stage {STAGE_NAME} completed <<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "MODEL_TRAINING"
try:
    logger.info(f"\n\n>>>>> Stage {STAGE_NAME} started <<<<<\n\n")
    pipe = ModelTrainingPipeline()
    pipe.run()
    logger.info(f"\n\n>>>>> Stage {STAGE_NAME} completed <<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e
