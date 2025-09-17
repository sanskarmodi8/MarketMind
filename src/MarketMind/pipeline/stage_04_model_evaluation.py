from MarketMind import logger
from MarketMind.components.model_evaluation import ModelEvaluation
from MarketMind.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL_EVALUATION"


class ModelEvaluationPipeline:
    """
    This class is responsible for managing the model evaluation stage of the pipeline.
    """

    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.model_evaluation = ModelEvaluation(config=self.config)

    def run(self):
        self.model_evaluation.run()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>> Stage {STAGE_NAME} started <<<<<\n\n")
        pipe = ModelEvaluationPipeline()
        pipe.run()
        logger.info(f"\n\n>>>>> Stage {STAGE_NAME} completed <<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
