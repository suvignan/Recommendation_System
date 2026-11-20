"""Runs the training pipeline end to end: `python main.py`."""
from recsys.exception import RecsysException
from recsys.logger import logger
from recsys.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        TrainingPipeline().run()
    except RecsysException as error:
        logger.error(f"Pipeline stopped: {error}")
        raise
