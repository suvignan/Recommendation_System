"""Day 15 smoke test: the WHOLE pipeline on the tiny committed fixture (tests/fixtures), with MLflow off.

The run itself happens once in conftest.py (`fixture_run`); these tests check what it produced.
It checks the PLUMBING (every stage runs and hands correct files to the next), not quality.
"""
import json
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from recsys.utils.common import sha256_of

FIXTURE = Path(__file__).parent / "fixtures"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_all_stages_run_on_the_fixture(fixture_run):
    output = fixture_run

    # 1. The table: 20 customers with history (fixture customers 0-19), each with 10 different real articles
    table = pd.read_parquet(output / "recommendation_table" / "recommendations.parquet")
    real_articles = set(pd.read_parquet(FIXTURE / "raw" / "articles.parquet")["article_id"])
    lists = table.groupby("customer_join_idx")["article_id"].apply(list)
    assert len(lists) == 20
    for articles in lists:
        assert len(articles) == 10
        assert len(set(articles)) == 10                                           # no article twice
        assert set(articles) <= real_articles                                     # never ALS's empty column 0
    assert len(pd.read_parquet(output / "recommendation_table" / "customer_lookup.parquet")) == 24

    # 2. The gate report exists, it passed, and the cold path really ran
    report = read_json(output / "model_evaluation" / "report_w09.json")
    assert report["gate"]["passed"] is True
    assert report["customers"]["cold"] > 0

    # 3. The table's checksums match the files on disk
    receipt = read_json(output / "recommendation_table" / "artifact.json")
    for name in ("recommendations", "bucket_lists", "customer_lookup"):
        assert sha256_of(receipt[f"{name}_path"]) == receipt[f"{name}_sha256"], name

    # 4. The serving files: the right shapes, and they match their manifest checksums
    serving = output / "recommendation_table" / "serving"
    manifest = read_json(serving / "serving_manifest.json")
    assert manifest["shapes"]["recs.npy"] == [21, 10]                             # 20 customers + empty row 0
    assert manifest["shapes"]["lookup_ids.npy"] == [24]
    for name, checksum in manifest["sha256"].items():
        assert sha256_of(serving / name) == checksum, name


def test_missing_day_fails_at_validation(tmp_path, pipeline):
    # Break the fixture on purpose: copy it, then delete every purchase on one day (2020-08-05)
    broken = tmp_path / "raw"
    broken.mkdir()
    for name in ("customers", "articles"):
        (broken / f"{name}.parquet").write_bytes((FIXTURE / "raw" / f"{name}.parquet").read_bytes())
    transactions = pq.read_table(FIXTURE / "raw" / "transactions_train.parquet")
    without_that_day = transactions.filter(pc.not_equal(transactions["t_dat"], "2020-08-05"))
    pq.write_table(without_that_day, broken / "transactions_train.parquet")

    output = tmp_path / "artifacts"
    finished = pipeline(broken, output)

    assert finished.returncode != 0                                               # the run stopped
    assert "Data validation failed" in finished.stderr                            # with a clear message
    report = read_json(output / "data_validation" / "report.json")
    assert report["failed"] == ["dates"]                                          # exactly the right check
    assert not (output / "data_splitter").exists()                                # and nothing after it ran
    assert not (output / "recommendation_table").exists()
