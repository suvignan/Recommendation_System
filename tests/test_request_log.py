"""Day 16 Step 5: logging served recommendations to Neon must never break serving, and must never store raw ids."""
import secrets
from pathlib import Path

import psycopg
import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from app import main
from app.request_log import RequestLogger, hash_customer

ROOT = Path(__file__).parents[1]
ENV = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}


@pytest.fixture
def serving(fixture_run):
    return fixture_run / "recommendation_table" / "serving"


def start_app(folder, logger, monkeypatch):
    monkeypatch.setattr(main, "SERVING_FOLDER", folder)
    monkeypatch.setattr(main, "state", main.ServingState())
    monkeypatch.setattr(main, "request_log", logger)
    return TestClient(main.app)


def test_hash_is_not_the_raw_id_and_needs_the_key():
    customer_id = "a" * 64
    assert hash_customer(customer_id, "key-1") != customer_id
    assert hash_customer(customer_id, "key-1") == hash_customer(customer_id, "key-1")      # same customer, same hash
    assert hash_customer(customer_id, "key-1") != hash_customer(customer_id, "key-2")      # without the key: no match


def test_logging_off_when_not_configured():
    assert RequestLogger(None, None).enabled is False
    assert RequestLogger("postgresql://x", None).enabled is False                          # a key is required too


def test_a_failing_database_never_breaks_a_request(serving, monkeypatch):
    class BrokenDatabase(RequestLogger):
        def setup(self):
            pass                                       # pretend the startup worked...
        def write_row(self, *args):
            raise psycopg.OperationalError("database is down")    # ...then every write fails

    with start_app(serving, BrokenDatabase("postgresql://unused", "key"), monkeypatch) as client:
        response = client.get("/recommend", params={"customer_id": "f" * 64})
    assert response.status_code == 200                 # the customer still gets their list
    assert len(response.json()["articles"]) == 10


@pytest.mark.skipif(not (ENV.get("RECSYS_DATABASE_URL") and ENV.get("RECSYS_HASH_KEY")),
                    reason="needs the real Neon login in .env (scripts/setup_neon.py); never in CI")
def test_one_request_writes_one_row_to_neon(serving, monkeypatch):
    url, key = ENV["RECSYS_DATABASE_URL"], ENV["RECSYS_HASH_KEY"]
    customer_id = secrets.token_hex(32)                # a new random 64-hex id, so we can find exactly our row

    with start_app(serving, RequestLogger(url, key), monkeypatch) as client:
        response = client.get("/recommend", params={"customer_id": customer_id})
    assert response.status_code == 200

    with psycopg.connect(url) as connection:
        rows = connection.execute(
            "SELECT customer_hash, source, articles, table_version, lookup_ms "
            "FROM recsys.served_recommendations WHERE customer_hash = %s", (hash_customer(customer_id, key),)
        ).fetchall()
    assert len(rows) == 1                              # exactly one row for one request
    customer_hash, source, articles, table_version, lookup_ms = rows[0]
    assert customer_id not in str(rows[0])             # the raw id is nowhere in the row
    assert source == response.json()["source"]
    assert articles == response.json()["articles"]
    print("logged row:", customer_hash[:16] + "...", source, articles[:3], table_version, round(lookup_ms, 3))
