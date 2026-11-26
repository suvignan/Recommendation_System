"""Check a RUNNING API (the local Docker container, or the live Render URL) against the real serving files.

    python scripts/check_live.py http://localhost:8000
    python scripts/check_live.py https://<your-service>.onrender.com --latency 200

1. /ready and /version: ready, and the same table_version as the files on this machine.
2. Parity: 20 fixed customers (with history, without history, unknown) get exactly the answers that the app
   gives in-process (TestClient) on the same files.
3. --latency N: N requests from this machine; p50 / p95 of the full round trip vs the app's own X-Lookup-Ms.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                        # so `from app import main` works when run as a script
from app import main  # noqa: E402

TABLE = ROOT / "artifacts" / "recommendation_table"


def twenty_customers():
    lookup = pd.read_parquet(TABLE / "customer_lookup.parquet", columns=["customer_id", "has_history"])
    with_history = lookup[lookup["has_history"]].sample(8, random_state=7)["customer_id"].tolist()
    without = lookup[~lookup["has_history"]].sample(8, random_state=7)["customer_id"].tolist()
    unknown = ["f" * 64, "e" * 64, "d" * 64, "c" * 64]
    requests = [{"customer_id": c} for c in with_history + without + unknown[:2]]
    requests += [{"customer_id": unknown[2], "age": 25}, {"customer_id": unknown[3], "age": 7}]
    return requests


def main_check(base_url, latency_requests):
    remote = httpx.Client(base_url=base_url, timeout=60)

    ready = remote.get("/ready")
    print("/ready  :", ready.status_code, ready.json())
    manifest = json.loads((TABLE / "serving" / "serving_manifest.json").read_text(encoding="utf-8"))
    version = remote.get("/version").json()
    print("/version: same table_version as the local files:", version.get("table_version") == manifest["table_version"])

    # Parity: the remote app vs the same app in-process, on the same files
    main.SERVING_FOLDER = TABLE / "serving"
    with TestClient(main.app) as local:
        mismatched = 0
        sources = {}
        for params in twenty_customers():
            here = local.get("/recommend", params=params).json()
            there = remote.get("/recommend", params=params).json()
            sources[there["source"]] = sources.get(there["source"], 0) + 1
            if here != there:
                mismatched = mismatched + 1
        print(f"Parity : {20 - mismatched}/20 identical | sources seen: {sources}")

    if latency_requests:
        round_trip = []
        inside = []
        for number in range(latency_requests):
            params = {"customer_id": f"{number:064x}"}
            start = time.perf_counter()
            response = remote.get("/recommend", params=params)
            round_trip.append((time.perf_counter() - start) * 1000)
            inside.append(float(response.headers["X-Lookup-Ms"]))
        cuts = statistics.quantiles(round_trip, n=100)
        print(f"Latency over {latency_requests} requests: p50 {cuts[49]:.1f} ms, p95 {cuts[94]:.1f} ms "
              f"(round trip) | inside the app: median {statistics.median(inside):.3f} ms")


if __name__ == "__main__":
    url = sys.argv[1]
    count = int(sys.argv[sys.argv.index("--latency") + 1]) if "--latency" in sys.argv else 0
    main_check(url, count)
