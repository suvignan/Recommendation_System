"""Run the API image locally on port 8000, giving it ONLY the two variables the API needs (Day 16 Step 6).

    python scripts/run_local_container.py            # starts container "recsys-test" from image recsys-api:local

Not `docker run --env-file .env`: that would also hand the container the Neon OWNER login and the MLflow token,
which the API never needs. Values are read from .env and passed through the environment; they're never printed.
"""
import os
import subprocess
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
NEEDED = ["RECSYS_DATABASE_URL", "RECSYS_HASH_KEY"]

settings = dotenv_values(ROOT / ".env")
environment = os.environ.copy()
for name in NEEDED:
    if settings.get(name):
        environment[name] = settings[name]

subprocess.run(["docker", "rm", "-f", "recsys-test"], capture_output=True)
command = ["docker", "run", "-d", "--name", "recsys-test", "-p", "8000:8000"]
for name in NEEDED:
    command += ["-e", name]                     # "-e NAME" (no value): docker copies it from our environment
command.append("recsys-api:local")
subprocess.run(command, env=environment, check=True, capture_output=True)
print("Started recsys-test on http://localhost:8000 with", [name for name in NEEDED if name in environment])
