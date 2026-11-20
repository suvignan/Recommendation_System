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
        report_path=constants.VALIDATION_DIR / "report.json",
    )
