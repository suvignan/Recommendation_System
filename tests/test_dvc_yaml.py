"""Day 15 Step 4: dvc.yaml and the code must agree on where every stage writes.

If someone renames a folder in one place but not the other, DVC and the code silently disagree about where
files live (DVC would cache an empty folder while the code writes somewhere else). This catches it at once.
"""
from pathlib import Path

import yaml

from recsys import constants
from recsys.pipeline.training_pipeline import EVALUATION_WEEK, SERVING_WEEK, artifact_file

ROOT = Path(__file__).parents[1]
DVC = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]


def relative(path):
    return Path(path).resolve().relative_to(constants.ROOT).as_posix()


def test_every_dvc_out_is_where_the_code_writes_and_the_reverse():
    dvc_outs = {out for stage in DVC.values() for out in stage["outs"]}
    code_outs = {relative(artifact_file(stage).parent) for stage in ("ingest", "validate", "split", "evaluate", "table")}
    code_outs |= {relative(artifact_file(stage, week).parent)
                  for stage in ("lists", "train") for week in (EVALUATION_WEEK, SERVING_WEEK)}
    assert dvc_outs == code_outs, f"only in dvc.yaml: {dvc_outs - code_outs}; only in code: {code_outs - dvc_outs}"


def test_every_code_and_config_dependency_exists():
    # A renamed code file would otherwise just vanish from DVC's view (and stop triggering reruns)
    deps = {dep for stage in DVC.values() for dep in stage["deps"]
            if dep.startswith(("src/", "config/", "data/RAW/"))}
    missing = sorted(dep for dep in deps if not (ROOT / dep).exists() and not dep.startswith("data/RAW/"))
    assert missing == []
