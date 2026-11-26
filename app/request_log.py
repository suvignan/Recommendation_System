"""Day 16 Step 5: log every served recommendation to Neon (Postgres), in the project's own schema `recsys`.

Three rules:
1. Logging must NEVER break serving: the row is written AFTER the response is sent (FastAPI BackgroundTasks),
   and any database error becomes a warning, not a failed request.
2. No raw customer ids in the log: we store HMAC-SHA256(secret key, customer_id). Plain SHA-256 could be reversed
   by hashing all 1.37M known ids; without the key, the hashes can't be matched back to customers.
3. Off unless configured: no RECSYS_DATABASE_URL or RECSYS_HASH_KEY (tests, CI) -> logging is simply off.

The app connects as `recsys_app`, a role that can only use the `recsys` schema (scripts/setup_neon.py), so even a
bug here can't touch the other projects' tables. The app creates its own table if needed and never DROPs anything.
"""
import hashlib
import hmac
import logging
import os

import psycopg

logger = logging.getLogger("uvicorn.error")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS recsys.served_recommendations (
    id            BIGSERIAL PRIMARY KEY,
    served_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    customer_hash TEXT        NOT NULL,
    source        TEXT        NOT NULL,
    articles      INTEGER[]   NOT NULL,
    table_version TEXT        NOT NULL,
    lookup_ms     REAL        NOT NULL
)
"""

INSERT_ROW = """
INSERT INTO recsys.served_recommendations (customer_hash, source, articles, table_version, lookup_ms)
VALUES (%s, %s, %s, %s, %s)
"""


def hash_customer(customer_id, key):
    """HMAC-SHA256: the same customer always gives the same hash (so we can count repeat visits), but nobody
    without the key can work out which customer it is."""
    return hmac.new(key.encode(), customer_id.encode(), hashlib.sha256).hexdigest()


class RequestLogger:
    def __init__(self, database_url=None, hash_key=None):
        self.database_url = database_url
        self.hash_key = hash_key
        self.enabled = bool(database_url) and bool(hash_key)

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get("RECSYS_DATABASE_URL"), os.environ.get("RECSYS_HASH_KEY"))

    def setup(self):
        """At startup: create our table if it isn't there. If the database can't be reached, turn logging off
        (the API still serves)."""
        if not self.enabled:
            logger.info("Request logging: off (RECSYS_DATABASE_URL / RECSYS_HASH_KEY not set)")
            return
        try:
            with psycopg.connect(self.database_url) as connection:
                connection.execute(CREATE_TABLE)
            logger.info("Request logging: on (recsys.served_recommendations)")
        except Exception as error:
            self.enabled = False
            logger.warning(f"Request logging: off, database not reachable at startup ({type(error).__name__})")

    def log(self, customer_id, source, articles, table_version, lookup_ms):
        """Called by FastAPI AFTER the response was sent. Never raises: a logging problem is only a warning."""
        if not self.enabled:
            return
        try:
            self.write_row(hash_customer(customer_id, self.hash_key), source, articles, table_version, lookup_ms)
        except Exception as error:
            logger.warning(f"Request logging: one row skipped ({type(error).__name__}: {error})")

    def write_row(self, customer_hash, source, articles, table_version, lookup_ms):
        # One short connection per row: simple, and Neon's pooled host (-pooler) makes connecting cheap
        with psycopg.connect(self.database_url) as connection:
            connection.execute(INSERT_ROW, (customer_hash, source, articles, table_version, lookup_ms))
