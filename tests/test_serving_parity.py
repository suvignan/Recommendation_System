"""Day 16 Step 1 parity: the dense .npy serving files give EXACTLY the table's lists, for the same 1,000 customers
as the Day 14 table test (500 with history + 500 without), going through the API's own path:
customer_id -> 64-bit id -> np.searchsorted -> rec row / bucket -> one row read.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.components.model_evaluation import long_to_lists
from recsys.utils.shared import customer_id_to_uint64

ROOT = Path(__file__).parents[1]
TABLE = ROOT / "artifacts" / "recommendation_table"
SERVING = TABLE / "serving"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not ((SERVING / "recs.npy").exists() and (TABLE / "customer_lookup.parquet").exists()),
                       reason="needs the real table + serving files: run `python main.py` or `dvc pull`"),
]


def test_npy_lists_equal_table_lists():
    lookup = pd.read_parquet(TABLE / "customer_lookup.parquet")
    sample = pd.concat([lookup[lookup["has_history"]].sample(500, random_state=42),
                        lookup[~lookup["has_history"]].sample(500, random_state=42)])
    table_lists = long_to_lists(pd.read_parquet(TABLE / "recommendations.parquet"), "customer_join_idx")
    table_buckets = long_to_lists(pd.read_parquet(TABLE / "bucket_lists.parquet"), "age_bucket")

    recs = np.load(SERVING / "recs.npy")
    bucket_lists = np.load(SERVING / "bucket_lists.npy")
    ids = np.load(SERVING / "lookup_ids.npy")
    rec_rows = np.load(SERVING / "lookup_rec_row.npy")
    buckets = np.load(SERVING / "lookup_bucket.npy")

    mismatched = []
    for row in sample.itertuples():
        number = np.uint64(customer_id_to_uint64(row.customer_id))
        position = np.searchsorted(ids, number)
        assert ids[position] == number                                  # found, not just "where it would go"
        if rec_rows[position] > 0:
            served = recs[rec_rows[position]].tolist()
            expected = table_lists[row.customer_join_idx]
        else:
            served = bucket_lists[buckets[position]].tolist()
            expected = table_buckets[row.age_bucket]
        if served != list(expected):
            mismatched.append(row.customer_id)
    assert mismatched == [], f"{len(mismatched)} of 1000 differ"
