from MarketMind import logger
from MarketMind.components.model_training import ModelTraining
from MarketMind.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL_TRAINING"


class ModelTrainingPipeline:
    """
    Pipeline to handle the model training stage.
    """

    def __init__(self):
        self.config = ConfigurationManager().get_model_training_config()
        self.model_training = ModelTraining(config=self.config)

    def run(self):
        self.model_training.run()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>> Stage {STAGE_NAME} started <<<<<\n\n")
        pipe = ModelTrainingPipeline()
        pipe.run()
        logger.info(f"\n\n>>>>> Stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
