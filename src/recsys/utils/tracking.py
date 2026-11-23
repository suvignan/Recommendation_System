"""MLflow: one run per pipeline run, started by the evaluation stage; the table stage adds its checksum file to it.

Where it logs comes ONLY from environment variables (never from code or committed files):
  MLFLOW_TRACKING_URI       e.g. https://dagshub.com/<user>/<repo>.mlflow
  MLFLOW_TRACKING_USERNAME  your DagsHub user name
  MLFLOW_TRACKING_PASSWORD  a DagsHub token
If MLFLOW_TRACKING_URI isn't set, runs go to a local folder (mlruns/, git-ignored), so the code works offline.
"""
import json
import os
import subprocess

import mlflow

from recsys import constants
from recsys.logger import logger
from recsys.utils.common import read_yaml, save_json

EXPERIMENT = "hm-recsys-pipeline"
RUN_FILE = constants.MODEL_EVALUATION_DIR / "mlflow_run.json"     # tells the table stage which run to add to


def connect():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        uri = (constants.ROOT / "mlruns").as_uri()
        mlflow.set_tracking_uri(uri)
    if uri.startswith("file:"):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"     # MLflow 3 refuses a local folder store otherwise (offline only)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.get_tracking_uri()


def git_commit():
    """The commit this run's code came from, plus whether there were uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=constants.ROOT).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                    cwd=constants.ROOT).stdout.strip())
        return commit, dirty
    except OSError:
        return "unknown", True


def log_evaluation(report_path, serving_week):
    """Log params, metrics, tags and the report. Called after the evaluation stage, pass OR fail."""
    try:
        uri = connect()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        params = read_yaml(constants.PARAMS_FILE)
        commit, dirty = git_commit()

        with mlflow.start_run(run_name=f"pipeline w{report['history_end_week']:02d}->w{serving_week:02d}") as run:
            mlflow.log_params({f"als.{key}": value for key, value in params["als"].items()})
            mlflow.log_params({f"recommend.{key}": value for key, value in params["recommend"].items()})
            mlflow.log_params({f"gate.{key}": value for key, value in params["gate"].items()})
            mlflow.log_params({"evaluation_history_end_week": report["history_end_week"],
                               "serving_history_end_week": serving_week})

            metrics = {}
            for part in ("all", "warm", "cold"):
                metrics[f"recall_{part}"] = report["final_system"][part]["recall"]
                metrics[f"ndcg_{part}"] = report["final_system"][part]["ndcg"]
                metrics[f"popularity_recall_{part}"] = report["last_week_popularity"][part]["recall"]
            metrics["lift_over_popularity"] = metrics["recall_all"] / metrics["popularity_recall_all"]
            mlflow.log_metrics(metrics)

            mlflow.set_tags({"gate_passed": str(report["gate"]["passed"]), "git_commit": commit,
                             "git_dirty": str(dirty),
                             "warnings_flagged": str(sum(w["flagged"] for w in report["warnings"]))})
            mlflow.log_artifact(str(report_path))
            save_json({"run_id": run.info.run_id, "tracking_uri": uri}, RUN_FILE)
        logger.info(f"MLflow run {run.info.run_id} logged to {uri}")
    except Exception as error:                      # tracking must never break the pipeline itself
        logger.warning(f"MLflow logging skipped: {error}")


def log_table(artifact_json_path):
    """Add the table's checksum file (its artifact JSON) to the evaluation stage's run."""
    try:
        connect()
        run_id = json.loads(RUN_FILE.read_text(encoding="utf-8"))["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(artifact_json_path))
        logger.info(f"MLflow: table checksums added to run {run_id}")
    except Exception as error:
        logger.warning(f"MLflow logging skipped: {error}")
