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


@dataclass(frozen=True)
class DataValidationArtifact:
    report_path: Path              # JSON: one entry per check
    validation_passed: bool
