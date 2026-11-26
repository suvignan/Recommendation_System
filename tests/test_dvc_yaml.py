"""Day 15 Step 4: dvc.yaml and the code must agree on where every stage writes.

If a folder is renamed in one place but not the other, DVC and the code silently disagree about where files live.
"""
from pathlib import Path

import yaml

from recsys import constants
from recsys.pipeline.training_pipeline import EVALUATION_WEEK, SERVING_WEEK, artifact_file

ROOT = Path(__file__).parents[1]


def read_dvc_stages():
    with open(ROOT / "dvc.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["stages"]


def folder_as_text(path):
    """artifacts/... relative to the project, the way dvc.yaml writes it."""
    return Path(path).resolve().relative_to(constants.ROOT).as_posix()


def test_every_dvc_out_is_where_the_code_writes_and_the_reverse():
    # The folders dvc.yaml says each stage writes
    dvc_folders = set()
    for stage in read_dvc_stages().values():
        for out in stage["outs"]:
            dvc_folders.add(out)

    # The folders the code really writes (artifact_file is what the pipeline itself uses)
    code_folders = set()
    for stage in ("ingest", "validate", "split", "evaluate", "table"):
        code_folders.add(folder_as_text(artifact_file(stage).parent))
    for stage in ("lists", "train"):
        for week in (EVALUATION_WEEK, SERVING_WEEK):
            code_folders.add(folder_as_text(artifact_file(stage, week).parent))

    assert dvc_folders == code_folders, \
        f"only in dvc.yaml: {dvc_folders - code_folders}; only in code: {code_folders - dvc_folders}"


def test_every_code_and_config_dependency_exists():
    # A renamed code file would otherwise just disappear from DVC's view (and stop triggering reruns)
    missing = []
    for stage in read_dvc_stages().values():
        for dep in stage["deps"]:
            if dep.startswith(("src/", "config/")) and not (ROOT / dep).exists():
                missing.append(dep)
    assert missing == []
