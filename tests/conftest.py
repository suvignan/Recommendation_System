"""Shared test setup. pytest finds this file by its name and gives its fixtures to every test file.

`fixture_run` runs the whole pipeline on the tiny committed fixture ONCE per test session; the smoke test and
the API tests both read its output, so the 9 stages don't run twice.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures"


def run_pipeline(raw_folder, output_folder):
    """Run main.py on raw_folder, writing into output_folder. Returns the finished program (exit code + output).

    A separate program, told by environment variables to read the fixture instead of data/RAW, use
    params_fixture.yaml, and write into a temp folder instead of artifacts/ (constants.py reads these once,
    when Python starts, so they can't be changed from inside this already-running test).
    """
    settings = os.environ.copy()
    settings["RECSYS_RAW_DIR"] = str(raw_folder)                                  # read these raw files
    settings["RECSYS_PARAMS_FILE"] = str(FIXTURE / "params_fixture.yaml")         # with the fixture settings
    settings["RECSYS_ARTIFACT_DIR"] = str(output_folder)                          # write here, not artifacts/
    settings["RECSYS_TRACKING"] = "off"                                           # no MLflow
    return subprocess.run([sys.executable, "main.py"], cwd=ROOT, env=settings,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)


@pytest.fixture
def pipeline():
    """The run_pipeline function itself, for tests that run the pipeline on their own (e.g. broken) data."""
    return run_pipeline


@pytest.fixture(scope="session")
def fixture_run(tmp_path_factory):
    """The pipeline's output folder for the fixture data (built once, shared by every test that asks for it)."""
    output = tmp_path_factory.mktemp("fixture_run") / "artifacts"
    finished = run_pipeline(FIXTURE / "raw", output)
    assert finished.returncode == 0, finished.stderr[-3000:]                     # 0 = the program succeeded
    return output
