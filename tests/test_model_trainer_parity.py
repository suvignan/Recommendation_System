"""Day 14 Step 1 parity: the pipeline's ModelTrainer must reproduce notebook 05's ALS (models/als, weeks 0-9).

Needs `python main.py` to have run first; skips with a message if either side is missing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.components.model_trainer import build_map, build_matrix

ROOT = Path(__file__).parents[1]
PIPELINE = ROOT / "artifacts" / "model_trainer" / "w09"
NOTEBOOK = ROOT / "models" / "als"


def load_or_skip(path):
    if not path.exists():
        pytest.skip(f"{path} not found: run `python main.py` first (models/ is not in Git)")
    return np.load(path)


def test_matrix_single_cell_day7():
    # Day 7 check: customer 21 bought article 917434002 twice in weeks 0-9 -> proves the row/column ORDER is right
    train_path = ROOT / "artifacts" / "data_splitter" / "train.parquet"
    if not train_path.exists():
        pytest.skip(f"{train_path} not found: run `python main.py` first")
    history = pd.read_parquet(train_path, columns=['customer_join_idx', 'article_id'])
    customer_map = build_map(history['customer_join_idx'])
    article_map = build_map(history['article_id'])
    matrix = build_matrix(history, customer_map, article_map)
    assert matrix[customer_map[21], article_map[917434002]] == 2
    assert matrix.sum() == len(history)


@pytest.mark.parametrize("name", ["user_factors.npy", "item_factors.npy"])
def test_vectors_match_notebook(name):
    mine = load_or_skip(PIPELINE / name)
    notebook = load_or_skip(NOTEBOOK / name)
    assert mine.shape == notebook.shape
    print(name, "largest difference:", np.abs(mine - notebook).max())
    assert np.allclose(mine, notebook, atol=1e-5)
