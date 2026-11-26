from dataclasses import dataclass
from pathlib import Path

# What each stage PRODUCED: output paths plus the key numbers (a receipt the next stage and the logs can read).


@dataclass(frozen=True)
class DataIngestionArtifact:
    transactions_path: Path        # 12-week window, same-day duplicates removed, with a week column
    customers_path: Path           # customers with customer_join_idx, alias applied
    articles_path: Path
    customer_map_path: Path        # customer_id -> customer_join_idx (file order, as on Day 1)
    rows_in_window: int            # 3,386,963 for this snapshot
    rows_after_dedup: int          # 3,085,875 for this snapshot
    raw_bad_date_rows: int         # raw t_dat values NOT written 'YYYY-MM-DD' (the window filter would skip them)


@dataclass(frozen=True)
class DataValidationArtifact:
    report_path: Path              # JSON: one entry per check
    validation_passed: bool


@dataclass(frozen=True)
class DataSplitterArtifact:
    train_path: Path               # weeks 0-9
    val_path: Path                 # week 10
    test_path: Path                # week 11
    n_train: int
    n_val: int
    n_test: int


@dataclass(frozen=True)
class ListBuilderArtifact:
    repurchase_path: Path          # long format: customer_join_idx, rank, article_id (newest purchase = rank 1)
    last_week_path: Path           # long format: rank, article_id (top filler_candidates of the last history week)
    age_bucket_path: Path          # long format: age_bucket, rank, article_id; bucket 0 = unknown age = the global list
    history_end_week: int          # which moment these lists are for (9 = evaluation, 11 = serving)
    n_repurchase_customers: int

@dataclass(frozen=True)
class ModelTrainerArtifact:
    user_factors_path: Path          # .npy, shape (customers + 1, 64): row i = the customer numbered i (row 0 unused)
    item_factors_path: Path          # .npy, shape (articles + 1, 64): row j = the article numbered j
    customer_map_path: Path          # parquet value/index: customer_join_idx -> row number (SORTED ids -> 1..N)
    article_map_path: Path           # parquet value/index: article_id -> column number (SORTED ids -> 1..M)
    settings_path: Path              # JSON echo: settings, implicit version, weeks used, counts, train time
    n_customers: int                 # how many customers were used to train this model
    n_articles: int                  # how many articles were used to train this model
    train_seconds: float                # how long the training took


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    report_path: Path                # JSON: final system + popularity for all / warm / cold, and every gate check
    recall_all: float
    ndcg_all: float
    gate_passed: bool                # always True when returned: a failed gate raises (after writing the report)


@dataclass(frozen=True)
class RecommendationTableArtifact:
    recommendations_path: Path       # long format: customer_join_idx, rank, article_id; every customer with history
    bucket_lists_path: Path          # long format: age_bucket, rank, article_id; bucket 0 = unknown age = global list
    customer_lookup_path: Path       # customer_id -> customer_join_idx, age_bucket, has_history (ALL known customers)
    serving_manifest_path: Path      # serving/serving_manifest.json: the .npy files the API loads + their SHA-256s
    table_version: str               # one checksum over all serving files (the API reports it)
    history_end_week: int            # 11: the lists are for the week AFTER the data ends
    n_customers_with_lists: int
    n_recommendation_rows: int
    n_lookup_customers: int
    recommendations_sha256: str      # the API checks these at startup: never serve a half-written or mismatched file
    bucket_lists_sha256: str
    customer_lookup_sha256: str

