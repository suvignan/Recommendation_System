"""MLflow: ONE run per pipeline run, started by the evaluation stage; the table stage adds its checksums to it.

Where it logs comes only from environment variables (in .env, never committed):
  MLFLOW_TRACKING_URI       https://dagshub.com/<user>/<repo>.mlflow
  MLFLOW_TRACKING_USERNAME  your DagsHub user name
  MLFLOW_TRACKING_PASSWORD  a DagsHub token
No MLFLOW_TRACKING_URI -> runs go to a local folder (mlruns/, git-ignored).
RECSYS_TRACKING=off -> no MLflow at all (the smoke test and CI).
"""
import json
import os
import subprocess

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")      # stop MLflow 3 printing a tooling hint on import

import mlflow  # noqa: E402

from recsys import constants
from recsys.logger import logger
from recsys.utils.common import read_yaml, save_json

EXPERIMENT = "hm-recsys-pipeline"
RUN_FILE = constants.MODEL_EVALUATION_DIR / "mlflow_run.json"     # the run id, so the table stage can add to it


def tracking_off():
    return os.environ.get("RECSYS_TRACKING", "").lower() == "off"


def connect():
    """Point MLflow at DagsHub (from .env) or, if nothing is set, at the local mlruns/ folder."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if uri == "":
        uri = (constants.ROOT / "mlruns").as_uri()
        mlflow.set_tracking_uri(uri)
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"         # MLflow 3 refuses a local folder unless told it's OK
    mlflow.set_experiment(EXPERIMENT)
    return uri


def git_commit():
    """(commit hash, True if there are uncommitted changes): which code produced this run."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=constants.ROOT)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=constants.ROOT)
        return commit.stdout.strip(), status.stdout.strip() != ""
    except OSError:                                             # git not installed
        return "unknown", True


def log_evaluation(report_path, serving_week):
    """After the evaluation stage, PASS or FAIL: params, metrics, tags and the report go into one MLflow run."""
    if tracking_off():
        return
    try:
        uri = connect()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        params = read_yaml(constants.PARAMS_FILE)
        commit, dirty = git_commit()

        # Params: the settings that produced this result
        run_params = {
            "evaluation_history_end_week": report["history_end_week"],
            "serving_history_end_week": serving_week,
        }
        for section in ("als", "recommend", "gate"):
            for key, value in params[section].items():
                run_params[f"{section}.{key}"] = value              # e.g. "als.factors": 64

        # Metrics: the scores, for all / warm / cold customers
        metrics = {}
        for part in ("all", "warm", "cold"):
            metrics[f"recall_{part}"] = report["final_system"][part]["recall"]
            metrics[f"ndcg_{part}"] = report["final_system"][part]["ndcg"]
            metrics[f"popularity_recall_{part}"] = report["last_week_popularity"][part]["recall"]
        if metrics["popularity_recall_all"] > 0:
            metrics["lift_over_popularity"] = metrics["recall_all"] / metrics["popularity_recall_all"]

        # Tags: did the gate pass, and which code was it
        flagged = 0
        for warning in report["warnings"]:
            if warning["flagged"]:
                flagged = flagged + 1
        tags = {
            "gate_passed": str(report["gate"]["passed"]),
            "git_commit": commit,
            "git_dirty": str(dirty),
            "warnings_flagged": str(flagged),
        }

        run_name = f"pipeline w{report['history_end_week']:02d}->w{serving_week:02d}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(run_params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(tags)
            mlflow.log_artifact(str(report_path))
            save_json({"run_id": run.info.run_id, "tracking_uri": uri}, RUN_FILE)
        logger.info(f"MLflow run {run.info.run_id} logged to {uri}")

    except Exception as error:
        # On purpose: a tracking problem (no network, wrong token) must never stop the pipeline itself
        logger.warning(f"MLflow logging skipped: {error}")


def log_table(artifact_json_path):
    """After the table stage: add the table's checksum file to the SAME run the evaluation stage started."""
    if tracking_off():
        return
    try:
        connect()
        run_id = json.loads(RUN_FILE.read_text(encoding="utf-8"))["run_id"]
        with mlflow.start_run(run_id=run_id):                   # reopen that run instead of starting a new one
            mlflow.log_artifact(str(artifact_json_path))
        logger.info(f"MLflow: table checksums added to run {run_id}")
    except Exception as error:
        logger.warning(f"MLflow logging skipped: {error}")
