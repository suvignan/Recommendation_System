from dataclasses import dataclass
from datetime import date
from pathlib import Path

from recsys import constants
from recsys.entity.artifact_entity import (DataIngestionArtifact, DataSplitterArtifact, ListBuilderArtifact,
                                           ModelEvaluationArtifact, ModelTrainerArtifact)
from recsys.utils.common import read_yaml

# What each stage NEEDS. The get_..._config functions are the ONLY place that reads schema.yaml and params.yaml;
# components receive a ready config, so a test can build one by hand (tiny fixture files, no YAML).


@dataclass(frozen=True)
class DataIngestionConfig:
    transactions_file: Path
    customers_file: Path
    articles_file: Path
    transaction_columns: tuple[str, ...]     # tuples, not lists: a frozen dataclass can't stop list.append()
    customer_columns: tuple[str, ...]
    article_columns: tuple[str, ...]
    aliases: dict                            # {'fashion_news_frequency': {'None': 'NONE'}}
    window_start: date
    window_weeks: int
    dedup_key: tuple[str, ...]
    expected_window_rows: int | None         # None (YAML null) = live weekly run: don't check counts
    expected_after_dedup: int | None
    output_dir: Path


def get_ingestion_config(schema_file=constants.SCHEMA_FILE, params_file=constants.PARAMS_FILE) -> DataIngestionConfig:
    schema = read_yaml(schema_file)
    params = read_yaml(params_file)
    return DataIngestionConfig(
        transactions_file=constants.RAW_DIR / schema["transactions"]["file"],
        customers_file=constants.RAW_DIR / schema["customers"]["file"],
        articles_file=constants.RAW_DIR / schema["articles"]["file"],
        transaction_columns=tuple(schema["transactions"]["columns"]),      # a dict gives its keys
        customer_columns=tuple(schema["customers"]["columns"]),
        article_columns=tuple(schema["articles"]["columns"]),
        aliases=schema["customers"]["known_aliases"],
        window_start=date.fromisoformat(params["window"]["start_date"]),  # a typo fails HERE, early and clearly
        window_weeks=params["window"]["weeks"],
        dedup_key=tuple(params["dedup"]["key_columns"]),
        expected_window_rows=params["expected_rows"]["window"],
        expected_after_dedup=params["expected_rows"]["after_dedup"],
        output_dir=constants.INGESTION_DIR,
    )


@dataclass(frozen=True)
class DataValidationConfig:
    transactions_path: Path                  # ingestion's OUTPUT (from its artifact), not the raw files
    customers_path: Path
    articles_path: Path
    transaction_columns: dict                # name -> pandas dtype of ingestion's output (schema: ingested_transactions)
    customer_columns: dict                   # name -> Parquet dtype, from the schema
    article_columns: dict
    allowed_values: dict                     # {'club_member_status': [...], 'fashion_news_frequency': [...]}
    window_start: date
    window_weeks: int
    dedup_key: tuple[str, ...]
    expected_after_dedup: int | None
    raw_bad_date_rows: int                   # measured by ingestion over the whole raw t_dat column
    report_path: Path


def get_validation_config(ingestion_artifact: DataIngestionArtifact, schema_file=constants.SCHEMA_FILE,
                          params_file=constants.PARAMS_FILE) -> DataValidationConfig:
    schema = read_yaml(schema_file)
    params = read_yaml(params_file)
    return DataValidationConfig(
        transactions_path=ingestion_artifact.transactions_path,
        customers_path=ingestion_artifact.customers_path,
        articles_path=ingestion_artifact.articles_path,
        transaction_columns=schema["ingested_transactions"]["columns"],
        customer_columns=schema["customers"]["columns"],
        article_columns=schema["articles"]["columns"],
        allowed_values=schema["customers"]["allowed_values"],
        window_start=date.fromisoformat(params["window"]["start_date"]),
        window_weeks=params["window"]["weeks"],
        dedup_key=tuple(params["dedup"]["key_columns"]),
        expected_after_dedup=params["expected_rows"]["after_dedup"],
        raw_bad_date_rows=ingestion_artifact.raw_bad_date_rows,
        report_path=constants.VALIDATION_DIR / "report.json",
    )


@dataclass(frozen=True)
class DataSplitterConfig:
    transactions_path: Path                  # ingestion's output (already validated)
    train_first_week: int
    train_last_week: int
    val_week: int
    test_week: int
    expected_train: int | None               # None = live run: don't check counts
    expected_val: int | None
    expected_test: int | None
    output_dir: Path


def get_splitter_config(ingestion_artifact: DataIngestionArtifact, params_file=constants.PARAMS_FILE) -> DataSplitterConfig:
    params = read_yaml(params_file)
    first_week, last_week = params["split"]["train_weeks"]
    return DataSplitterConfig(
        transactions_path=ingestion_artifact.transactions_path,
        train_first_week=first_week,
        train_last_week=last_week,
        val_week=params["split"]["val_week"],
        test_week=params["split"]["test_week"],
        expected_train=params["expected_rows"]["train"],
        expected_val=params["expected_rows"]["val"],
        expected_test=params["expected_rows"]["test"],
        output_dir=constants.SPLIT_DIR,
    )


@dataclass(frozen=True)
class ListBuilderConfig:
    transactions_path: Path                  # ALL ingested weeks; history_end_week decides how much of it is used
    customers_path: Path                     # for the age bucket
    history_end_week: int                    # 9 = evaluation lists (scored on week 10), 11 = serving lists
    k: int                                   # 10 articles per list
    repurchase_max: int                      # at most 10 own purchases
    filler_candidates: int                   # the last-week list keeps the top 30
    age_edges: tuple[int, ...]
    output_dir: Path


def get_list_builder_config(ingestion_artifact: DataIngestionArtifact, history_end_week: int,
                            params_file=constants.PARAMS_FILE) -> ListBuilderConfig:
    params = read_yaml(params_file)
    return ListBuilderConfig(
        transactions_path=ingestion_artifact.transactions_path,
        customers_path=ingestion_artifact.customers_path,
        history_end_week=history_end_week,
        k=params["recommend"]["k"],
        repurchase_max=params["recommend"]["repurchase_max"],
        filler_candidates=params["recommend"]["filler_candidates"],
        age_edges=tuple(params["features"]["age_edges"]),
        output_dir=constants.LISTS_DIR / f"history_to_week_{history_end_week}",
    )

@dataclass(frozen=True)
class ModelTrainerConfig:
    transactions_path: Path
    history_end_week: int
    factors: int
    regularization: float
    iterations: int
    alpha: float
    random_state: int
    output_dir: Path
def get_model_trainer_config(ingestion_artifact: DataIngestionArtifact, history_end_week: int,
                             params_file=constants.PARAMS_FILE) -> ModelTrainerConfig:
    params = read_yaml(params_file)
    return ModelTrainerConfig(
        transactions_path=ingestion_artifact.transactions_path,
        history_end_week=history_end_week,
        factors=params["als"]["factors"],
        regularization=params["als"]["regularization"],
        iterations=params["als"]["iterations"],
        alpha=params["als"]["alpha"],
        random_state=params["als"]["random_state"],
        output_dir=constants.MODEL_TRAINER_DIR / f"w{history_end_week:02d}",      # 9 -> w09
    )


@dataclass(frozen=True)
class ModelEvaluationConfig:
    val_path: Path                   # the answer key: what each customer bought in the week AFTER the history
    customers_path: Path             # age -> bucket, for cold customers
    repurchase_path: Path            # the three list files from ListBuilder (long format)
    last_week_path: Path
    age_bucket_path: Path
    user_factors_path: Path          # the ALS parts from ModelTrainer
    item_factors_path: Path
    customer_map_path: Path
    article_map_path: Path
    history_end_week: int            # the lists and the model must both be built from weeks 0..this
    k: int                           # 10
    filler_candidates: int           # ALS top 30, so the list never comes up short after skipping own articles
    age_edges: tuple[int, ...]
    reference_recall: float | None   # snapshot parity; None in a live weekly run
    reference_tolerance: float
    min_lift_over_popularity: float  # live gate: system recall >= this x popularity recall (same customers, same run)
    cold_must_beat_popularity: bool
    report_path: Path


def get_evaluation_config(ingestion_artifact: DataIngestionArtifact, splitter_artifact: DataSplitterArtifact,
                          lists_artifact: ListBuilderArtifact, trainer_artifact: ModelTrainerArtifact,
                          params_file=constants.PARAMS_FILE) -> ModelEvaluationConfig:
    params = read_yaml(params_file)
    return ModelEvaluationConfig(
        val_path=splitter_artifact.val_path,
        customers_path=ingestion_artifact.customers_path,
        repurchase_path=lists_artifact.repurchase_path,
        last_week_path=lists_artifact.last_week_path,
        age_bucket_path=lists_artifact.age_bucket_path,
        user_factors_path=trainer_artifact.user_factors_path,
        item_factors_path=trainer_artifact.item_factors_path,
        customer_map_path=trainer_artifact.customer_map_path,
        article_map_path=trainer_artifact.article_map_path,
        history_end_week=lists_artifact.history_end_week,
        k=params["recommend"]["k"],
        filler_candidates=params["recommend"]["filler_candidates"],
        age_edges=tuple(params["features"]["age_edges"]),
        reference_recall=params["gate"]["reference_recall"],
        reference_tolerance=params["gate"]["reference_tolerance"],
        min_lift_over_popularity=params["gate"]["min_lift_over_popularity"],
        cold_must_beat_popularity=params["gate"]["cold_must_beat_popularity"],
        report_path=constants.MODEL_EVALUATION_DIR / f"report_w{lists_artifact.history_end_week:02d}.json",
    )


@dataclass(frozen=True)
class RecommendationTableConfig:
    gate_passed: bool                # from the evaluation artifact: a recipe that failed evaluation never builds a table
    customers_path: Path             # ingestion: customer_id, customer_join_idx, age (ALL customers, buyers or not)
    repurchase_path: Path            # serving lists (ListBuilder, history_end_week = serving week)
    age_bucket_path: Path
    user_factors_path: Path          # serving ALS (ModelTrainer, same history)
    item_factors_path: Path
    customer_map_path: Path
    article_map_path: Path
    history_end_week: int
    k: int
    filler_candidates: int
    age_edges: tuple[int, ...]
    output_dir: Path


def get_table_config(ingestion_artifact: DataIngestionArtifact, evaluation_artifact: ModelEvaluationArtifact,
                     lists_artifact: ListBuilderArtifact, trainer_artifact: ModelTrainerArtifact,
                     params_file=constants.PARAMS_FILE) -> RecommendationTableConfig:
    params = read_yaml(params_file)
    return RecommendationTableConfig(
        gate_passed=evaluation_artifact.gate_passed,
        customers_path=ingestion_artifact.customers_path,
        repurchase_path=lists_artifact.repurchase_path,
        age_bucket_path=lists_artifact.age_bucket_path,
        user_factors_path=trainer_artifact.user_factors_path,
        item_factors_path=trainer_artifact.item_factors_path,
        customer_map_path=trainer_artifact.customer_map_path,
        article_map_path=trainer_artifact.article_map_path,
        history_end_week=lists_artifact.history_end_week,
        k=params["recommend"]["k"],
        filler_candidates=params["recommend"]["filler_candidates"],
        age_edges=tuple(params["features"]["age_edges"]),
        output_dir=constants.RECOMMENDATION_TABLE_DIR,
    )
