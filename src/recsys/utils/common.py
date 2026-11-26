import json
from pathlib import Path

import yaml

from recsys import constants
from recsys.utils.shared import sha256_of  # noqa: F401  (one copy, shared with the API)


def read_yaml(path) -> dict:
    """Load a YAML file. safe_load only: yaml.load can execute arbitrary code."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path) -> None:
    """Write a dict as readable JSON, creating the folder if needed. default=str turns Paths and dates into text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def path_to_text(path) -> str:
    """A path as text for a JSON file. Inside the project -> 'artifacts/...' (still right after a fresh clone in
    another folder). Outside the project (a test's temp folder) -> the full path."""
    path = Path(path).resolve()
    try:
        return path.relative_to(constants.ROOT).as_posix()
    except ValueError:                           # relative_to raises ValueError when path is not inside ROOT
        return path.as_posix()


def save_artifact(artifact, path) -> None:
    """Save a stage's artifact (its receipt) as JSON, so the NEXT stage can run as a separate command (DVC)."""
    data = {}
    for name, value in vars(artifact).items():   # vars() gives the dataclass fields as {name: value}
        if isinstance(value, Path):
            value = path_to_text(value)
        data[name] = value
    save_json(data, path)


def load_artifact(artifact_class, path):
    """Read a JSON written by save_artifact back into the artifact class (text paths become Paths again)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, value in data.items():
        if name.endswith("_path"):
            data[name] = constants.ROOT / value  # ROOT / a full path = that full path, so both kinds work
    return artifact_class(**data)
