"""Run ONE stage from the command line (this is what dvc.yaml calls):

    python -m recsys.pipeline.run --stage ingest
    python -m recsys.pipeline.run --stage lists --history-end-week 9
"""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')      # FIRST, before numpy loads: same ALS vectors every run

import sys  # noqa: E402

# Windows consoles default to a code page that can't print emoji (MLflow prints one): UTF-8 instead
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from recsys.exception import RecsysException  # noqa: E402
from recsys.logger import logger  # noqa: E402
from recsys.pipeline.training_pipeline import STAGES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run one pipeline stage")
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--history-end-week", type=int, help="lists / train only: 9 = evaluation, 11 = serving")
    args = parser.parse_args()

    load_dotenv()                                        # MLflow credentials from .env (never committed)
    try:
        if args.stage in ("lists", "train"):
            if args.history_end_week is None:
                parser.error(f"--stage {args.stage} needs --history-end-week")
            STAGES[args.stage](args.history_end_week)
        else:
            STAGES[args.stage]()
    except RecsysException as error:
        logger.error(f"Stage {args.stage} stopped: {error}")
        raise


if __name__ == "__main__":
    main()
