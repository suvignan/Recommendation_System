"""Builds the tiny fake raw dataset the smoke test and CI run on (Day 15). Run: python tests/fixtures/make_fixture.py

Same files, columns and dtypes as H&M's raw data (t_dat as TEXT 'YYYY-MM-DD'), so every stage runs unchanged.
Deterministic: a fixed seed and fixed rules, so the committed output can always be regenerated identically.

What's inside, and why:
- 24 customers, 15 articles; at least 2 transactions on EVERY day of the 12-week window (validation checks
  that each week has 7 dates, so a gap would fail, correctly).
- Each week's 14 purchases cover 14 different articles, so every last-week / age-bucket list can be filled to 10.
- customers 0-15:  buy from week 0 (warm everywhere)
- customers 16-17: first buy in week 10 -> cold in evaluation (16 has NO age -> bucket 0 = the global list)
- customers 18-19: first buy in week 11 -> cold in the test week, warm in the serving table
- customers 20-23: never buy -> only in the customer lookup (the API's "known but no history" path)
- customer 5 uses the "None" spelling (the schema alias), customer 7 has a missing news frequency
- customer 3 buys the same article twice on one day (removed by dedup)
- 3 rows BEFORE the window (removed by the window filter)
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).parent / "raw"
START = pd.Timestamp("2020-07-01")          # params.yaml window.start_date (a Wednesday)
DAYS = 12 * 7
N_CUSTOMERS, N_ARTICLES = 24, 15
rng = np.random.default_rng(15)


def customer_id(i):
    return hashlib.sha256(f"fixture-customer-{i}".encode()).hexdigest()       # 64 hex characters, like H&M


ARTICLE_IDS = [700_000_001 + 1000 * i for i in range(N_ARTICLES)]


def transactions():
    rows = []
    for day in range(DAYS):
        week = day // 7
        date = (START + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        for slot in range(2):                                                   # 2 rows a day, warm pool 0-15
            rows.append((date, customer_id((day * 3 + slot) % 16), ARTICLE_IDS[(day * 2 + slot) % N_ARTICLES]))
        if week == 10:                                                          # new in week 10 (cold in evaluation)
            for customer in (16, 17):
                rows.append((date, customer_id(customer), ARTICLE_IDS[(day + customer) % N_ARTICLES]))
        if week == 11:                                                          # new in week 11
            for customer in (18, 19):
                rows.append((date, customer_id(customer), ARTICLE_IDS[(day + customer) % N_ARTICLES]))
    rows.append(("2020-07-08", customer_id(3), ARTICLE_IDS[4]))                # same-day duplicate...
    rows.append(("2020-07-08", customer_id(3), ARTICLE_IDS[4]))                # ...twice: dedup keeps one
    for day in (-6, -4, -2):                                                    # before the window: filtered out
        rows.append(((START + pd.Timedelta(days=day)).strftime("%Y-%m-%d"), customer_id(0), ARTICLE_IDS[0]))

    table = pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"])
    table["price"] = rng.uniform(0.005, 0.1, len(table)).round(4)
    table["sales_channel_id"] = rng.integers(1, 3, len(table))
    return table.astype({"t_dat": str, "customer_id": str, "article_id": "int64",
                         "price": "float64", "sales_channel_id": "int64"})


def customers():
    table = pd.DataFrame({
        "customer_id": [customer_id(i) for i in range(N_CUSTOMERS)],
        "FN": [1.0 if i % 3 == 0 else np.nan for i in range(N_CUSTOMERS)],
        "Active": [1.0 if i % 4 == 0 else np.nan for i in range(N_CUSTOMERS)],
        "club_member_status": [["ACTIVE", "PRE-CREATE", "LEFT CLUB"][i % 3] for i in range(N_CUSTOMERS)],
        "fashion_news_frequency": [["NONE", "Regularly", "Monthly"][i % 3] for i in range(N_CUSTOMERS)],
        "age": [float(18 + (i * 7) % 60) for i in range(N_CUSTOMERS)],
        "postal_code": [hashlib.sha256(f"postal-{i % 5}".encode()).hexdigest() for i in range(N_CUSTOMERS)],
    })
    table.loc[5, "fashion_news_frequency"] = "None"       # the alias spelling (schema known_aliases)
    table.loc[7, "fashion_news_frequency"] = None         # missing is allowed
    table.loc[16, "age"] = np.nan                         # no age -> bucket 0 (cold in evaluation)
    return table.astype({"customer_id": str, "club_member_status": str, "postal_code": str})


def articles():
    groups = ["Garment Upper body", "Garment Lower body", "Accessories"]
    return pd.DataFrame({
        "article_id": pd.Series(ARTICLE_IDS, dtype="int64"),
        "product_code": pd.Series([a // 1000 for a in ARTICLE_IDS], dtype="int64"),
        "prod_name": [f"Fixture item {i}" for i in range(N_ARTICLES)],
        "product_type_no": pd.Series([250 + i % 4 for i in range(N_ARTICLES)], dtype="int64"),
        "product_type_name": [["Sweater", "Trousers", "Dress", "Socks"][i % 4] for i in range(N_ARTICLES)],
        "product_group_name": [groups[i % 3] for i in range(N_ARTICLES)],
        "graphical_appearance_name": ["Solid"] * N_ARTICLES,
        "colour_group_name": [["Black", "White", "Blue"][i % 3] for i in range(N_ARTICLES)],
        "perceived_colour_master_name": [["Black", "White", "Blue"][i % 3] for i in range(N_ARTICLES)],
        "department_no": pd.Series([1600 + i % 5 for i in range(N_ARTICLES)], dtype="int64"),
        "index_group_name": ["Ladieswear"] * N_ARTICLES,
        "section_name": [["Womens Basics", "Divided"][i % 2] for i in range(N_ARTICLES)],
        "garment_group_name": [["Jersey Basic", "Trousers", "Knitwear"][i % 3] for i in range(N_ARTICLES)],
    })


def write_like_hm(table, path):
    """pandas 3 writes text as Parquet large_string; H&M's raw files use string. Store exactly what H&M stores."""
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    schema = pa.schema([pa.field(f.name, pa.string() if f.type == pa.large_string() else f.type) for f in arrow.schema])
    pq.write_table(arrow.cast(schema), path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in [("transactions_train", transactions()), ("customers", customers()), ("articles", articles())]:
        write_like_hm(table, OUT / f"{name}.parquet")
        print(f"{name}: {len(table)} rows, {(OUT / f'{name}.parquet').stat().st_size / 1024:.1f} KB")
