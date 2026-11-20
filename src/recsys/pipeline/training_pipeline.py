import time

from recsys.components.data_ingestion import DataIngestion
from recsys.components.data_splitter import DataSplitter
from recsys.components.data_validation import DataValidation
from recsys.components.list_builder import ListBuilder
from recsys.entity.config_entity import (get_ingestion_config, get_list_builder_config, get_splitter_config,
                                         get_validation_config)
from recsys.logger import logger
from recsys.utils.common import read_yaml
from recsys import constants


class TrainingPipeline:
    """Runs the stages in order. Each stage's artifact feeds the next stage's config. (Day 14 adds the model stages.)"""

    def run_stage(self, number, name, stage):
        logger.info(f">>> Stage {number}: {name}")
        stage_start = time.perf_counter()
        artifact = stage.run()
        logger.info(f"<<< Stage {number} done in {time.perf_counter() - stage_start:.1f}s")
        return artifact

    def run(self):
        pipeline_start = time.perf_counter()
        evaluation_week = read_yaml(constants.PARAMS_FILE)["lists"]["evaluation_history_end_week"]

        ingestion = self.run_stage(1, "data ingestion", DataIngestion(get_ingestion_config()))
        validation = self.run_stage(2, "data validation", DataValidation(get_validation_config(ingestion)))
        splits = self.run_stage(3, "data splitter", DataSplitter(get_splitter_config(ingestion)))
        lists = self.run_stage(4, f"list builder (evaluation history: weeks 0-{evaluation_week})",
                               ListBuilder(get_list_builder_config(ingestion, evaluation_week)))

        logger.info(f"Pipeline finished in {time.perf_counter() - pipeline_start:.1f}s")
        return ingestion, validation, splits, lists
