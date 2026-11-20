"""Golden-file parity: the pipeline's ListBuilder must reproduce notebook 07's lists exactly (history weeks 0-9).

The golden files are notebook 07's trusted output (the code that produced 0.0495), exported once to tests/golden/.
These tests compare the pipeline's output with them cell by cell. They need the pipeline to have run first
(`python main.py`), so they skip, with a message, if its output isn't there.
"""
from pathlib import Path

import pandas as pd
import pytest

GOLDEN = Path(__file__).parent / "golden"
PIPELINE = Path(__file__).parents[1] / "artifacts" / "list_builder" / "history_to_week_9"


def compare(pipeline_file, golden_file, sort_by):
    pipeline_path = PIPELINE / pipeline_file
    if not pipeline_path.exists():
        pytest.skip(f"{pipeline_path} not found: run `python main.py` first")
    # Sort by key + RANK (never by article): sorting by article would hide an ORDER difference, which NDCG cares about
    mine = pd.read_parquet(pipeline_path).sort_values(sort_by).reset_index(drop=True)
    golden = pd.read_parquet(GOLDEN / golden_file).sort_values(sort_by).reset_index(drop=True)
    pd.testing.assert_frame_equal(mine, golden)


def test_repurchase_lists_match_notebook():
    compare("repurchase.parquet", "golden_repurchase_w9.parquet", ["customer_join_idx", "rank"])


def test_last_week_list_matches_notebook():
    compare("last_week.parquet", "golden_last_week_w9.parquet", ["rank"])


def test_age_bucket_lists_match_notebook():
    compare("age_buckets.parquet", "golden_age_buckets_w9.parquet", ["age_bucket", "rank"])
