"""Day 15 end-to-end smoke test: ALL 9 stages on the tiny committed fixture, in tmp_path, with MLflow off.

It checks the PLUMBING (every stage runs and hands correct files to the next), not quality (see params_fixture.yaml).
Runs `python main.py` in a subprocess: the same entry point as a real run, with the paths redirected by env vars.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from recsys.utils.common import sha256_of

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures"


def run_pipeline(tmp_path, raw_dir):
    env = {
        **os.environ,
        "RECSYS_RAW_DIR": str(raw_dir),
        "RECSYS_PARAMS_FILE": str(FIXTURE / "params_fixture.yaml"),
        "RECSYS_ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "RECSYS_TRACKING": "off",                         # no MLflow: no credentials, no network
    }
    result = subprocess.run([sys.executable, "main.py"], cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=300)
    return result, tmp_path / "artifacts"


def test_all_nine_stages_run_on_the_fixture(tmp_path):
    result, artifacts = run_pipeline(tmp_path, FIXTURE / "raw")
    assert result.returncode == 0, result.stderr[-3000:]

    # The table: every customer with history (fixture customers 0-19) has exactly 10 distinct real articles
    table = pd.read_parquet(artifacts / "recommendation_table" / "recommendations.parquet")
    per_customer = table.groupby("customer_join_idx")["article_id"].agg(["size", "nunique"])
    assert len(per_customer) == 20
    assert (per_customer["size"] == 10).all() and (per_customer["nunique"] == 10).all()
    real_articles = set(pd.read_parquet(FIXTURE / "raw" / "articles.parquet")["article_id"])
    assert set(table["article_id"]) <= real_articles                 # e.g. never ALS's empty column 0
    assert len(pd.read_parquet(artifacts / "recommendation_table" / "customer_lookup.parquet")) == 24

    # The gate report exists and passed
    report = json.loads((artifacts / "model_evaluation" / "report_w09.json").read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is True
    assert report["customers"]["cold"] > 0                          # the cold path really ran

    # The checksums in the table's artifact JSON match the files on disk
    manifest = json.loads((artifacts / "recommendation_table" / "artifact.json").read_text(encoding="utf-8"))
    for name in ("recommendations", "bucket_lists", "customer_lookup"):
        assert sha256_of(manifest[f"{name}_path"]) == manifest[f"{name}_sha256"], name


def test_missing_day_fails_at_validation(tmp_path):
    # Break the fixture on purpose: delete every purchase on one day (Wednesday 2020-08-05, week 5)
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("customers", "articles"):
        (raw / f"{name}.parquet").write_bytes((FIXTURE / "raw" / f"{name}.parquet").read_bytes())
    transactions = pq.read_table(FIXTURE / "raw" / "transactions_train.parquet")
    pq.write_table(transactions.filter(pc.not_equal(transactions["t_dat"], "2020-08-05")),
                   raw / "transactions_train.parquet")

    result, artifacts = run_pipeline(tmp_path, raw)

    assert result.returncode != 0
    assert "Data validation failed" in result.stderr                 # a clear message, not a confusing crash later
    report = json.loads((artifacts / "data_validation" / "report.json").read_text(encoding="utf-8"))
    assert report["failed"] == ["dates"]                             # exactly the right check, nothing else
    assert not (artifacts / "data_splitter").exists()                # nothing after validation ran
    assert not (artifacts / "recommendation_table").exists()
