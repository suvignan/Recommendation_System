"""Day 14 Step 4 parity: what the table SERVES == what the frozen system computes LIVE, for a fixed sample.

Served = the Day 16 request logic on the saved files: customer_id -> lookup -> personal list, else bucket list.
Live   = recomputed now from the raw history (weeks 0-11) + the serving ALS, like final_system_recommend on Day 11.
Needs `python main.py` (or `dvc repro`) to have run first; skips otherwise.
"""
from pathlib import Path

import pandas as pd
import pytest

from recsys.components.list_builder import age_bucket_top10, age_to_bucket, fill_up, last_week_top, repurchase_lists
from recsys.components.model_evaluation import load_als_parts, long_to_lists, warm_lists
from recsys.utils.common import read_yaml

ROOT = Path(__file__).parents[1]
TABLE = ROOT / "artifacts" / "recommendation_table"
INGESTION = ROOT / "artifacts" / "data_ingestion"
TRAINER = ROOT / "artifacts" / "model_trainer" / "w11"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not ((TABLE / "recommendations.parquet").exists() and (TRAINER / "user_factors.npy").exists()),
                       reason="needs the real table and serving ALS: run `python main.py` or `dvc pull`"),
]


def test_served_lists_equal_live_recipe():
    params = read_yaml(ROOT / "params.yaml")
    k, filler = params["recommend"]["k"], params["recommend"]["filler_candidates"]
    edges = params["features"]["age_edges"]
    week = params["lists"]["serving_history_end_week"]

    # ---- the fixed sample: 500 customers with history + 500 without (both request paths)
    lookup = pd.read_parquet(TABLE / "customer_lookup.parquet")
    sample = pd.concat([lookup[lookup["has_history"]].sample(500, random_state=42),
                        lookup[~lookup["has_history"]].sample(500, random_state=42)])

    # ---- SERVED: only the three saved files, exactly what the API will do
    recommendations = long_to_lists(pd.read_parquet(TABLE / "recommendations.parquet"), "customer_join_idx")
    buckets_served = long_to_lists(pd.read_parquet(TABLE / "bucket_lists.parquet"), "age_bucket")
    served = {}
    for row in sample.itertuples():
        if row.customer_join_idx in recommendations:
            served[row.customer_id] = recommendations[row.customer_join_idx]
        else:
            served[row.customer_id] = buckets_served[row.age_bucket]

    # ---- LIVE: recomputed from the raw history, no saved lists
    transactions = pd.read_parquet(INGESTION / "transactions.parquet")
    history = transactions[transactions["week"] <= week]
    customers = pd.read_parquet(INGESTION / "customers.parquet", columns=["customer_id", "customer_join_idx", "age"])
    own_lists = repurchase_lists(history, k)
    global_top = last_week_top(history, week, k)
    bucket_tops = age_bucket_top10(history, customers, week, edges, k)
    als_parts = load_als_parts(TRAINER / "user_factors.npy", TRAINER / "item_factors.npy",
                               TRAINER / "customer_map.parquet", TRAINER / "article_map.parquet")

    sample_customers = customers[customers["customer_id"].isin(sample["customer_id"])].set_index("customer_id")
    warm_ids = [idx for idx in sample_customers["customer_join_idx"] if idx in own_lists]
    warm = warm_lists(warm_ids, own_lists, als_parts, k, filler)
    live = {}
    for customer_id, row in sample_customers.iterrows():
        if row["customer_join_idx"] in own_lists:
            live[customer_id] = warm[row["customer_join_idx"]]
        else:
            bucket = age_to_bucket(pd.Series([row["age"]]), edges).iloc[0]
            live[customer_id] = fill_up(bucket_tops.get(bucket, []), global_top, k)

    assert len(served) == len(live) == 1000
    mismatched = [customer_id for customer_id in served if list(served[customer_id]) != list(live[customer_id])]
    assert mismatched == [], f"{len(mismatched)} of 1000 customers differ, e.g. {mismatched[:3]}"
