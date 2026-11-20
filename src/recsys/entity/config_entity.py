from dataclasses import dataclass
from datetime import date
from pathlib import Path

from recsys import constants
from recsys.entity.artifact_entity import DataIngestionArtifact
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
