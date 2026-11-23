"""Runs the whole pipeline end to end: `python main.py` (DVC runs the same stages one by one: `dvc repro`)."""
import os

# FIRST, before anything imports numpy: one BLAS thread, so ALS gives the same vectors every run (Day 7)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys  # noqa: E402

# Windows consoles default to a code page that can't print emoji (MLflow prints one): UTF-8 instead
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv  # noqa: E402

from recsys.exception import RecsysException  # noqa: E402
from recsys.logger import logger  # noqa: E402
from recsys.pipeline.training_pipeline import run_all  # noqa: E402

if __name__ == "__main__":
    load_dotenv()                  # MLflow credentials from .env (never committed)
    try:
        run_all()
    except RecsysException as error:
        logger.error(f"Pipeline stopped: {error}")
        raise
