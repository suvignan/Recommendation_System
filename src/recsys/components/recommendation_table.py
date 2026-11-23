import time

import pandas as pd

from recsys.components.list_builder import age_to_bucket
from recsys.components.model_evaluation import load_als_parts, long_to_lists, warm_lists
from recsys.entity.artifact_entity import RecommendationTableArtifact
from recsys.entity.config_entity import RecommendationTableConfig
from recsys.exception import RecsysException
from recsys.logger import logger
from recsys.utils.common import sha256_of


class RecommendationTable:
    """The weekly output the API serves: every buyer's final top k, the age-bucket lists, and a customer lookup.
    Built from the SERVING history (weeks 0-11) with the SAME recipe the gate just passed on weeks 0-9."""

    def __init__(self, config: RecommendationTableConfig):
        self.config = config

    def run(self) -> RecommendationTableArtifact:
        try:
            c = self.config
            start_time = time.perf_counter()
            if not c.gate_passed:
                raise ValueError("The evaluation gate did not pass: refusing to build a serving table")

            # 1. Every customer with history: own purchases first, then ALS fills (the frozen warm path)
            own_lists = long_to_lists(pd.read_parquet(c.repurchase_path), 'customer_join_idx')
            warm_ids = sorted(own_lists)
            als_parts = load_als_parts(c.user_factors_path, c.item_factors_path, c.customer_map_path, c.article_map_path)
            lists = warm_lists(warm_ids, own_lists, als_parts, c.k, c.filler_candidates)

            rows = [(customer, rank, article)
                    for customer in warm_ids
                    for rank, article in enumerate(lists[customer], start=1)]
            recommendations = pd.DataFrame(rows, columns=['customer_join_idx', 'rank', 'article_id']).astype('int32')

            # Checks BEFORE saving: one row per (customer, rank), ranks 1..k, k distinct articles per customer
            per_customer = recommendations.groupby('customer_join_idx')['article_id'].agg(['size', 'nunique'])
            if not ((per_customer['size'] == c.k) & (per_customer['nunique'] == c.k)).all():
                raise ValueError(f"Some customers don't have exactly {c.k} distinct articles")
            if recommendations.duplicated(['customer_join_idx', 'rank']).any():
                raise ValueError("A customer has the same rank twice")

            # 2. The age-bucket lists (already filled to k by ListBuilder; bucket 0 = global)
            bucket_lists = pd.read_parquet(c.age_bucket_path)
            if len(bucket_lists) != (len(c.age_edges)) * c.k:
                raise ValueError(f"Expected {len(c.age_edges) * c.k} bucket rows, got {len(bucket_lists)}")

            # 3. The customer lookup: H&M's text id -> our index, age bucket, has history (buyers AND non-buyers)
            customers = pd.read_parquet(c.customers_path, columns=['customer_id', 'customer_join_idx', 'age'])
            lookup = pd.DataFrame({
                'customer_id': customers['customer_id'],
                'customer_join_idx': customers['customer_join_idx'].astype('int32'),
                'age_bucket': age_to_bucket(customers['age'], c.age_edges).astype('int8'),
                'has_history': customers['customer_join_idx'].isin(own_lists),
            })
            if lookup['customer_id'].duplicated().any():
                raise ValueError("Duplicate customer_id in the lookup")
            if lookup['has_history'].sum() != len(warm_ids):
                raise ValueError("Some customers with lists are missing from the customers file")

            c.output_dir.mkdir(parents=True, exist_ok=True)
            recommendations_path = c.output_dir / 'recommendations.parquet'
            bucket_lists_path = c.output_dir / 'bucket_lists.parquet'
            customer_lookup_path = c.output_dir / 'customer_lookup.parquet'
            recommendations.to_parquet(recommendations_path, index=False)
            bucket_lists.to_parquet(bucket_lists_path, index=False)
            lookup.to_parquet(customer_lookup_path, index=False)

            logger.info(f"Recommendations: {len(warm_ids):,} customers, {len(recommendations):,} rows")
            logger.info(f"Bucket lists: {len(bucket_lists)} rows | customer lookup: {len(lookup):,} customers "
                        f"({int(lookup['has_history'].sum()):,} with history)")
            logger.info(f"Recommendation table finished in {time.perf_counter() - start_time:.1f}s -> {c.output_dir}")

            return RecommendationTableArtifact(
                recommendations_path=recommendations_path,
                bucket_lists_path=bucket_lists_path,
                customer_lookup_path=customer_lookup_path,
                history_end_week=c.history_end_week,
                n_customers_with_lists=len(warm_ids),
                n_recommendation_rows=len(recommendations),
                n_lookup_customers=len(lookup),
                recommendations_sha256=sha256_of(recommendations_path),
                bucket_lists_sha256=sha256_of(bucket_lists_path),
                customer_lookup_sha256=sha256_of(customer_lookup_path),
            )

        except Exception as error:
            raise RecsysException(error) from error
