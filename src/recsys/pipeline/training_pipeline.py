import time

from recsys.components.data_ingestion import DataIngestion
from recsys.components.data_validation import DataValidation
from recsys.entity.config_entity import get_ingestion_config, get_validation_config
from recsys.logger import logger


class TrainingPipeline:
    """Runs the stages in order. Each stage's artifact feeds the next stage's config. (Days 13-14 add the rest.)"""

    def run(self):
        pipeline_start = time.perf_counter()

        logger.info(">>> Stage 1: data ingestion")
        stage_start = time.perf_counter()
        ingestion_artifact = DataIngestion(get_ingestion_config()).run()
        logger.info(f"<<< Stage 1 done in {time.perf_counter() - stage_start:.1f}s")

        logger.info(">>> Stage 2: data validation")
        stage_start = time.perf_counter()
        validation_artifact = DataValidation(get_validation_config(ingestion_artifact)).run()
        logger.info(f"<<< Stage 2 done in {time.perf_counter() - stage_start:.1f}s")

        logger.info(f"Pipeline finished in {time.perf_counter() - pipeline_start:.1f}s")
        return ingestion_artifact, validation_artifact
