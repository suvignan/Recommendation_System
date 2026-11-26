"""Day 16 Step 4: the API (app/main.py) against its contract (app/CONTRACT.md), one test per rule.

The app is pointed at the serving files the FIXTURE pipeline built (conftest.py `fixture_run`), so these tests
need no real data and run in CI. TestClient sends requests to the app directly, without starting a server.
"""
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main

UNKNOWN_ID = "f" * 64                                   # well-formed, but not a fixture customer


@pytest.fixture
def serving(fixture_run):
    return fixture_run / "recommendation_table" / "serving"


@pytest.fixture
def table(fixture_run):
    return fixture_run / "recommendation_table"


def start_app(folder, monkeypatch):
    """A fresh app reading `folder`. Use it with `with ... as client:` (that runs the startup = lifespan)."""
    monkeypatch.setattr(main, "SERVING_FOLDER", folder)
    monkeypatch.setattr(main, "state", main.ServingState())        # forget anything an earlier test loaded
    return TestClient(main.app)


def list_of(long_table, key_name, key):
    rows = long_table[long_table[key_name] == key].sort_values("rank")
    return rows["article_id"].tolist()


def test_customer_with_history_gets_their_own_list(serving, table, monkeypatch):
    lookup = pd.read_parquet(table / "customer_lookup.parquet")
    customer = lookup[lookup["has_history"]].iloc[0]
    expected = list_of(pd.read_parquet(table / "recommendations.parquet"), "customer_join_idx", customer["customer_join_idx"])

    with start_app(serving, monkeypatch) as client:
        response = client.get("/recommend", params={"customer_id": customer["customer_id"]})

    assert response.status_code == 200
    assert response.json()["source"] == "personal"
    assert response.json()["articles"] == expected
    assert float(response.headers["X-Lookup-Ms"]) >= 0                # the timing header is there


def test_never_buyer_gets_their_stored_age_bucket(serving, table, monkeypatch):
    lookup = pd.read_parquet(table / "customer_lookup.parquet")
    customer = lookup[~lookup["has_history"] & (lookup["age_bucket"] > 0)].iloc[0]     # fixture customers 20-23
    expected = list_of(pd.read_parquet(table / "bucket_lists.parquet"), "age_bucket", customer["age_bucket"])

    with start_app(serving, monkeypatch) as client:
        response = client.get("/recommend", params={"customer_id": customer["customer_id"]})

    assert response.json()["source"] == "age_bucket"
    assert response.json()["articles"] == expected


def test_unknown_customer_with_age_25_gets_the_25_to_29_bucket(serving, table, monkeypatch):
    expected = list_of(pd.read_parquet(table / "bucket_lists.parquet"), "age_bucket", 3)    # [25, 30) = bucket 3

    with start_app(serving, monkeypatch) as client:
        response = client.get("/recommend", params={"customer_id": UNKNOWN_ID, "age": 25})

    assert response.json()["source"] == "age_bucket"
    assert response.json()["articles"] == expected


def test_unknown_customer_without_age_gets_global(serving, table, monkeypatch):
    expected = list_of(pd.read_parquet(table / "bucket_lists.parquet"), "age_bucket", 0)    # bucket 0 = global

    with start_app(serving, monkeypatch) as client:
        response = client.get("/recommend", params={"customer_id": UNKNOWN_ID})

    assert response.json()["source"] == "global"
    assert response.json()["articles"] == expected


def test_bad_input_decisions(serving, monkeypatch):
    with start_app(serving, monkeypatch) as client:
        assert client.get("/recommend", params={"customer_id": "abc"}).status_code == 422           # too short
        assert client.get("/recommend", params={"customer_id": "g" * 64}).status_code == 422        # not hex
        assert client.get("/recommend", params={"customer_id": UNKNOWN_ID, "age": 150}).status_code == 422
        assert client.get("/recommend", params={"customer_id": UNKNOWN_ID, "age": "abc"}).status_code == 422
        young = client.get("/recommend", params={"customer_id": UNKNOWN_ID, "age": 7})
        assert young.status_code == 200 and young.json()["source"] == "global"                     # under 16
        upper = client.get("/recommend", params={"customer_id": UNKNOWN_ID.upper()})
        assert upper.status_code == 200                                                             # upper case OK


def test_tampered_file_is_never_served(serving, tmp_path, monkeypatch):
    # Copy the serving files, then change ONE byte at the end of recs.npy
    tampered = tmp_path / "serving"
    shutil.copytree(serving, tampered)
    data = bytearray((tampered / "recs.npy").read_bytes())
    data[-1] = (data[-1] + 1) % 256
    (tampered / "recs.npy").write_bytes(bytes(data))

    with start_app(tampered, monkeypatch) as client:
        assert client.get("/health").status_code == 200                          # alive...
        ready = client.get("/ready")
        assert ready.status_code == 503                                           # ...but not ready
        assert "recs.npy" in ready.json()["reason"]                               # and it says which file
        assert client.get("/recommend", params={"customer_id": UNKNOWN_ID}).status_code == 503


def test_version_returns_the_manifest_checksum(serving, monkeypatch):
    with start_app(serving, monkeypatch) as client:
        version = client.get("/version").json()
        ready = client.get("/ready").json()

    manifest = (serving / "serving_manifest.json").read_text(encoding="utf-8")
    assert version["table_version"] in manifest
    assert ready["table_version"] == version["table_version"][:12]


# ---- POST /recommend: the form version (both fields optional, the id travels in the body)

def test_post_with_customer_id_gives_the_same_answer_as_get(serving, table, monkeypatch):
    lookup = pd.read_parquet(table / "customer_lookup.parquet")
    customer_id = lookup[lookup["has_history"]].iloc[0]["customer_id"]
    with start_app(serving, monkeypatch) as client:
        by_get = client.get("/recommend", params={"customer_id": customer_id}).json()
        by_post = client.post("/recommend", json={"customer_id": customer_id}).json()
    assert by_post == by_get                                            # one shared serve(): same rules, same list
    assert by_post["source"] == "personal"


def test_post_with_only_an_age_gives_that_age_group(serving, table, monkeypatch):
    expected = list_of(pd.read_parquet(table / "bucket_lists.parquet"), "age_bucket", 3)     # 25 -> bucket 3
    with start_app(serving, monkeypatch) as client:
        response = client.post("/recommend", json={"age": 25})
    assert response.status_code == 200
    assert response.json() == {"customer_id": None, "source": "age_bucket", "articles": expected,
                               "table_version": response.json()["table_version"]}


def test_post_with_an_empty_form_gives_global(serving, monkeypatch):
    with start_app(serving, monkeypatch) as client:
        response = client.post("/recommend", json={})
    assert response.status_code == 200
    assert response.json()["source"] == "global"


def test_post_bad_input_is_422_like_get(serving, monkeypatch):
    with start_app(serving, monkeypatch) as client:
        assert client.post("/recommend", json={"customer_id": "abc"}).status_code == 422
        assert client.post("/recommend", json={"age": 150}).status_code == 422
        assert client.post("/recommend", json={"age": "abc"}).status_code == 422        # Pydantic checks the type
