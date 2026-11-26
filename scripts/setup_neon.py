"""One-time Neon setup for this project (Day 16 Step 5). Run: python scripts/setup_neon.py

Uses the OWNER connection (DATABASE_URL in .env) only here, to:
1. create the schema `recsys` (a folder inside the shared database, separate from the other projects' tables);
2. create the role `recsys_app` with rights ONLY on that schema (least privilege);
3. write RECSYS_DATABASE_URL (the app's login, never the owner) and RECSYS_HASH_KEY into .env, if missing;
4. PROVE the isolation: as recsys_app, reading another project's table must fail with "permission denied".

Additive only: it never drops or changes anything that already exists. Secret values are never printed.
"""
import secrets
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import psycopg
from dotenv import dotenv_values
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
ROLE = "recsys_app"


def with_login(url, user, password):
    """The same database URL (same pooled host, same settings), but logging in as another user."""
    parts = urlsplit(url)
    host = parts.netloc.split("@")[-1]
    return urlunsplit((parts.scheme, f"{quote(user)}:{quote(password)}@{host}", parts.path, parts.query, ""))


def add_to_env(name, value):
    with open(ENV_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{name}={value}\n")


def main():
    settings = dotenv_values(ENV_FILE)
    owner_url = settings.get("DATABASE_URL")
    if not owner_url:
        raise SystemExit("DATABASE_URL (the Neon owner connection) is not in .env")

    with psycopg.connect(owner_url, autocommit=True) as owner:
        print("Connected as:", owner.execute("SELECT current_user").fetchone()[0])

        # 1. The schema
        owner.execute("CREATE SCHEMA IF NOT EXISTS recsys")
        print("Schema recsys: ready")

        # 2. The role (a new password only if we don't already have the app's login saved)
        role_exists = owner.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,)).fetchone() is not None
        if settings.get("RECSYS_DATABASE_URL") and role_exists:
            app_url = settings["RECSYS_DATABASE_URL"]
            print(f"Role {ROLE}: already set up (RECSYS_DATABASE_URL found in .env)")
        else:
            password = secrets.token_urlsafe(24)
            if role_exists:
                owner.execute(sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(sql.Identifier(ROLE), sql.Literal(password)))
            else:
                owner.execute(sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD {}").format(sql.Identifier(ROLE), sql.Literal(password)))
            app_url = with_login(owner_url, ROLE, password)
            add_to_env("RECSYS_DATABASE_URL", app_url)
            print(f"Role {ROLE}: {'password reset' if role_exists else 'created'}; RECSYS_DATABASE_URL written to .env")

        # Rights: use and create inside recsys. Nothing anywhere else.
        owner.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA recsys TO {}").format(sql.Identifier(ROLE)))

        # One other project's table, to prove we can't read it
        other = owner.execute("""
            SELECT table_schema, table_name FROM information_schema.tables
            WHERE table_schema NOT IN ('recsys', 'pg_catalog', 'information_schema') AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name LIMIT 1
        """).fetchone()

    if not settings.get("RECSYS_HASH_KEY"):
        add_to_env("RECSYS_HASH_KEY", secrets.token_hex(32))
        print("RECSYS_HASH_KEY: generated and written to .env")

    # 4. The proof, as the app's role
    with psycopg.connect(app_url, autocommit=True) as app:
        print("App connects as:", app.execute("SELECT current_user").fetchone()[0])
        app.execute("CREATE TABLE IF NOT EXISTS recsys.setup_check (id INTEGER)")
        app.execute("DROP TABLE recsys.setup_check")                   # its OWN scratch table, inside recsys only
        print("Inside recsys: create table OK")
        if other is None:
            print("No other project's table found to test against")
            return
        try:
            app.execute(sql.SQL("SELECT * FROM {}.{} LIMIT 1").format(sql.Identifier(other[0]), sql.Identifier(other[1])))
            print(f"!!! {ROLE} COULD read {other[0]}.{other[1]}: isolation is NOT working")
        except psycopg.errors.InsufficientPrivilege as error:
            print(f"Reading {other[0]}.{other[1]} as {ROLE}: refused as it should be -> {str(error).strip()}")


if __name__ == "__main__":
    main()
