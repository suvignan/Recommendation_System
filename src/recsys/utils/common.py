import json
from pathlib import Path

import yaml


def read_yaml(path) -> dict:
    """Load a YAML file. safe_load only: yaml.load can execute arbitrary code."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path) -> None:
    """Write a dict as readable JSON, creating the folder if needed. default=str turns Paths and dates into text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
