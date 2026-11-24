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
TRAIN = ROOT / "artifacts" / "data_splitter" / "train.parquet"

pytestmark = pytest.mark.data
needs_train = pytest.mark.skipif(not TRAIN.exists(), reason="needs the real split: run `python main.py` or `dvc pull`")
needs_vectors = pytest.mark.skipif(not ((PIPELINE / "user_factors.npy").exists() and (NOTEBOOK / "user_factors.npy").exists()),
                                   reason="needs the real trained vectors AND notebook 05's models/als (not in Git)")


@needs_train
def test_matrix_single_cell_day7():
    # Day 7 check: customer 21 bought article 917434002 twice in weeks 0-9 -> proves the row/column ORDER is right
    history = pd.read_parquet(TRAIN, columns=['customer_join_idx', 'article_id'])
    customer_map = build_map(history['customer_join_idx'])
    article_map = build_map(history['article_id'])
    matrix = build_matrix(history, customer_map, article_map)
    assert matrix[customer_map[21], article_map[917434002]] == 2
    assert matrix.sum() == len(history)


@needs_vectors
@pytest.mark.parametrize("name", ["user_factors.npy", "item_factors.npy"])
def test_vectors_match_notebook(name):
    mine = np.load(PIPELINE / name)
    notebook = np.load(NOTEBOOK / name)
    assert mine.shape == notebook.shape
    print(name, "largest difference:", np.abs(mine - notebook).max())
    assert np.allclose(mine, notebook, atol=1e-5)
