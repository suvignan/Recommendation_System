"""The stages. Each one READS its inputs' artifact JSON from disk and WRITES its own, so it can run as a separate
command (DVC: python -m recsys.pipeline.run --stage ...). main.py runs them all in order with the same functions."""
import time

from recsys import constants
from recsys.components.data_ingestion import DataIngestion
from recsys.components.data_splitter import DataSplitter
from recsys.components.data_validation import DataValidation
from recsys.components.list_builder import ListBuilder
from recsys.components.model_evaluation import ModelEvaluation
from recsys.components.model_trainer import ModelTrainer
from recsys.components.recommendation_table import RecommendationTable
from recsys.entity.artifact_entity import (DataIngestionArtifact, DataSplitterArtifact, DataValidationArtifact,
                                           ListBuilderArtifact, ModelEvaluationArtifact, ModelTrainerArtifact)
from recsys.entity.config_entity import (get_evaluation_config, get_ingestion_config, get_list_builder_config,
                                         get_model_trainer_config, get_splitter_config, get_table_config,
                                         get_validation_config)
from recsys.exception import RecsysException
from recsys.logger import logger
from recsys.utils import tracking
from recsys.utils.common import load_artifact, read_yaml, save_artifact

PARAMS = read_yaml(constants.PARAMS_FILE)
EVALUATION_WEEK = PARAMS["lists"]["evaluation_history_end_week"]      # 9
SERVING_WEEK = PARAMS["lists"]["serving_history_end_week"]            # 11


def artifact_file(stage, week=None):
    """Where each stage's artifact JSON lives (inside its own output folder, so DVC tracks it with the outputs)."""
    folders = {
        "ingest": constants.INGESTION_DIR,
        "validate": constants.VALIDATION_DIR,
        "split": constants.SPLIT_DIR,
        "lists": constants.LISTS_DIR / f"history_to_week_{week}",
        "train": constants.MODEL_TRAINER_DIR / f"w{week:02d}" if week is not None else None,
        "evaluate": constants.MODEL_EVALUATION_DIR,
        "table": constants.RECOMMENDATION_TABLE_DIR,
    }
    return folders[stage] / "artifact.json"


def timed(name, stage):
    logger.info(f">>> {name}")
    start = time.perf_counter()
    artifact = stage.run()
    logger.info(f"<<< {name} done in {time.perf_counter() - start:.1f}s")
    return artifact


def require_gate():
    """The serving branch (weeks 0-11) only runs after the evaluation gate passed on weeks 0-9."""
    path = artifact_file("evaluate")
    if not path.exists() or not load_artifact(ModelEvaluationArtifact, path).gate_passed:
        raise RecsysException(ValueError(f"No passed evaluation gate ({path}): run the evaluate stage first"))


def ingest():
    artifact = timed("ingest", DataIngestion(get_ingestion_config()))
    save_artifact(artifact, artifact_file("ingest"))


def validate():
    ingestion = load_artifact(DataIngestionArtifact, artifact_file("ingest"))
    artifact = timed("validate", DataValidation(get_validation_config(ingestion)))
    save_artifact(artifact, artifact_file("validate"))


def split():
    load_artifact(DataValidationArtifact, artifact_file("validate"))      # only exists if validation passed
    ingestion = load_artifact(DataIngestionArtifact, artifact_file("ingest"))
    artifact = timed("split", DataSplitter(get_splitter_config(ingestion)))
    save_artifact(artifact, artifact_file("split"))


def lists(week):
    if week == SERVING_WEEK:
        require_gate()
    ingestion = load_artifact(DataIngestionArtifact, artifact_file("ingest"))
    artifact = timed(f"lists (history weeks 0-{week})", ListBuilder(get_list_builder_config(ingestion, week)))
    save_artifact(artifact, artifact_file("lists", week))


def train(week):
    if week == SERVING_WEEK:
        require_gate()
    ingestion = load_artifact(DataIngestionArtifact, artifact_file("ingest"))
    artifact = timed(f"train (history weeks 0-{week})", ModelTrainer(get_model_trainer_config(ingestion, week)))
    save_artifact(artifact, artifact_file("train", week))


def evaluate():
    ingestion = load_artifact(DataIngestionArtifact, artifact_file("ingest"))
    splits = load_artifact(DataSplitterArtifact, artifact_file("split"))
    week_lists = load_artifact(ListBuilderArtifact, artifact_file("lists", EVALUATION_WEEK))
    trainer = load_artifact(ModelTrainerArtifact, artifact_file("train", EVALUATION_WEEK))
    config = get_evaluation_config(ingestion, splits, week_lists, trainer)

    artifact_file("evaluate").unlink(missing_ok=True)      # an old "passed" receipt must not survive a failed run
    try:
        artifact = timed("evaluate (the gate)", ModelEvaluation(config))
    finally:
        if config.report_path.exists():                     # pass OR fail: the run is logged, tagged gate_passed
            tracking.log_evaluation(config.report_path, SERVING_WEEK)
    save_artifact(artifact, artifact_file("evaluate"))


def table():
    require_gate()
    ingestion = load_artifact(DataIngestionArtifact, artifact_file("ingest"))
    evaluation = load_artifact(ModelEvaluationArtifact, artifact_file("evaluate"))
    week_lists = load_artifact(ListBuilderArtifact, artifact_file("lists", SERVING_WEEK))
    trainer = load_artifact(ModelTrainerArtifact, artifact_file("train", SERVING_WEEK))
    artifact = timed("table", RecommendationTable(get_table_config(ingestion, evaluation, week_lists, trainer)))
    save_artifact(artifact, artifact_file("table"))
    tracking.log_table(artifact_file("table"))



STAGES = {
    "ingest": ingest,
    "validate": validate,
    "split": split,
    "lists": lists,               # needs --history-end-week
    "train": train,               # needs --history-end-week
    "evaluate": evaluate,
    "table": table,
}


def run_all():
    """The whole weekly run, in dvc.yaml's order. Stops at the first stage that raises (validation, gate, ...)."""
    start = time.perf_counter()
    ingest()
    validate()
    split()
    lists(EVALUATION_WEEK)
    train(EVALUATION_WEEK)
    evaluate()                    # the gate: if it fails, nothing below runs
    lists(SERVING_WEEK)
    train(SERVING_WEEK)
    table()
    logger.info(f"Pipeline finished in {time.perf_counter() - start:.1f}s")
