import time

import implicit
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from recsys.entity.artifact_entity import ModelTrainerArtifact
from recsys.entity.config_entity import ModelTrainerConfig
from recsys.exception import RecsysException
from recsys.logger import logger
from recsys.utils.common import save_json


def build_map(values):
    """{id: number}: SORTED ids -> 1..N (0 = unknown, left empty), exactly like Day 5's vocab."""
    return {
        value: number
        for number, value in enumerate(sorted(values.unique()), start=1)
    }


def build_matrix(history, customer_map, article_map):
    """Times bought per (customer, article): row = customer number, column = article number. Shape (N+1, M+1)."""
    grouped = (
        history.groupby(['customer_join_idx', 'article_id'])
        .size()
        .reset_index(name='times_bought')
    )
    rows = grouped['customer_join_idx'].map(customer_map).to_numpy()
    cols = grouped['article_id'].map(article_map).to_numpy()
    values = grouped['times_bought'].astype('float32').to_numpy()     # float32, like notebook 08 cell 19

    matrix = csr_matrix(
        (values, (rows, cols)),                                          # ONE tuple: (data, (rows, cols))
        shape=(len(customer_map) + 1, len(article_map) + 1)
    )
    return matrix


class ModelTrainer:
    """Trains ALS on weeks 0..history_end_week with the frozen settings; saves vectors + both mappings."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def run(self) -> ModelTrainerArtifact:
        try:
            c = self.config
            start_time = time.perf_counter()

            history = pd.read_parquet(c.transactions_path, columns=['customer_join_idx', 'article_id', 'week'])
            history = history[history['week'] <= c.history_end_week]
            logger.info(f"Training history: weeks {history['week'].min()}-{history['week'].max()}, {len(history):,} rows")

            customer_map = build_map(history['customer_join_idx'])
            article_map = build_map(history['article_id'])
            logger.info(f"Customer map: {len(customer_map):,} | article map: {len(article_map):,}")

            matrix = build_matrix(history, customer_map, article_map)
            logger.info(f"Matrix: {matrix.shape[0]:,} x {matrix.shape[1]:,}, {matrix.nnz:,} filled cells, "
                        f"sum {int(matrix.sum()):,} (expected {len(history):,})")

            model = AlternatingLeastSquares(
                factors=c.factors,
                regularization=c.regularization,
                iterations=c.iterations,
                alpha=c.alpha,
                random_state=c.random_state,
                use_gpu=False,
            )
            fit_start = time.perf_counter()
            model.fit(matrix, show_progress=False)
            train_seconds = round(time.perf_counter() - fit_start, 1)
            logger.info(f"ALS trained in {train_seconds}s")

            c.output_dir.mkdir(parents=True, exist_ok=True)
            user_factors_path = c.output_dir / "user_factors.npy"
            item_factors_path = c.output_dir / "item_factors.npy"
            customer_map_path = c.output_dir / "customer_map.parquet"
            article_map_path = c.output_dir / "article_map.parquet"
            settings_path = c.output_dir / "settings.json"

            np.save(user_factors_path, model.user_factors)
            np.save(item_factors_path, model.item_factors)
            for mapping, path in [(customer_map, customer_map_path), (article_map, article_map_path)]:
                pd.DataFrame({'value': list(mapping.keys()), 'index': list(mapping.values())}).to_parquet(path, index=False)

            save_json({
                'factors': c.factors,
                'iterations': c.iterations,
                'regularization': c.regularization,
                'alpha': c.alpha,
                'random_state': c.random_state,
                'implicit_version': implicit.__version__,
                'trained_on': f"weeks 0-{c.history_end_week}",
                'customers': len(customer_map),
                'articles': len(article_map),
                'train_seconds': train_seconds,
            }, settings_path)

            logger.info(f"Model trainer finished in {time.perf_counter() - start_time:.1f}s -> {c.output_dir}")
            return ModelTrainerArtifact(
                user_factors_path=user_factors_path,
                item_factors_path=item_factors_path,
                customer_map_path=customer_map_path,
                article_map_path=article_map_path,
                settings_path=settings_path,
                n_customers=len(customer_map),
                n_articles=len(article_map),
                train_seconds=train_seconds,
            )

        except Exception as error:
            raise RecsysException(error) from error
