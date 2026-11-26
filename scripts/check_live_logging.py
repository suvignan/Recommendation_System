"""Prove the LIVE service logs to Neon: send one request with a brand-new random id, then find exactly that row.

    python scripts/check_live_logging.py https://recommendation-system-bl3v.onrender.com

Reads the hash key and the app's login from .env (never printed). The row is found by its HMAC hash, so this
also proves Render uses the same RECSYS_HASH_KEY as your laptop.
"""
import secrets
import sys
import time
from pathlib import Path

import httpx
import psycopg
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.request_log import hash_customer  # noqa: E402

settings = dotenv_values(ROOT / ".env")
customer_id = secrets.token_hex(32)                        # 64 random hex characters: nobody has this id

response = httpx.get(f"{sys.argv[1]}/recommend", params={"customer_id": customer_id}, timeout=120)
print("live request:", response.status_code, response.json().get("source"))
time.sleep(3)                                              # the row is written just AFTER the response

with psycopg.connect(settings["RECSYS_DATABASE_URL"]) as connection:   # read back as recsys_app
    print("checking as:", connection.execute("SELECT current_user").fetchone()[0])
    rows = connection.execute(
        "SELECT served_at, source, table_version FROM recsys.served_recommendations WHERE customer_hash = %s",
        (hash_customer(customer_id, settings["RECSYS_HASH_KEY"]),)).fetchall()
print("rows for this request:", len(rows), rows)
print("PASS: exactly one row" if len(rows) == 1 else "FAIL: expected exactly one row")
