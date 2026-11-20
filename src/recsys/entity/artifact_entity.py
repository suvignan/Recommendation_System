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
