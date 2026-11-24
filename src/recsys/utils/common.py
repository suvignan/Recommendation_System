import dataclasses
import hashlib
import json
from pathlib import Path

import yaml

from recsys import constants


def read_yaml(path) -> dict:
    """Load a YAML file. safe_load only: yaml.load can execute arbitrary code."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path) -> None:
    """Write a dict as readable JSON, creating the folder if needed. default=str turns Paths and dates into text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def sha256_of(path) -> str:
    """The file's fingerprint (64 hex characters): change one byte and it changes completely. Read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifact(artifact, path) -> None:
    """A stage's artifact (dataclass) -> JSON, so the NEXT stage can run as a separate command (DVC).
    Paths are saved RELATIVE to the project root, so the file still works after a fresh clone + dvc pull elsewhere."""
    data = {}
    for name, value in dataclasses.asdict(artifact).items():
        if isinstance(value, Path):
            value = value.resolve()
            value = value.relative_to(constants.ROOT).as_posix() if value.is_relative_to(constants.ROOT) else value.as_posix()
        data[name] = value
    save_json(data, path)


def load_artifact(artifact_class, path):
    """JSON written by save_artifact -> the artifact dataclass (fields ending in _path become absolute Paths again;
    ROOT / an absolute path is just that absolute path)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in data:
        if name.endswith("_path") and data[name] is not None:
            data[name] = constants.ROOT / data[name]
    return artifact_class(**data)
