import hashlib
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from recsys.components.list_builder import age_to_bucket
from recsys.components.model_evaluation import load_als_parts, long_to_lists, warm_lists
from recsys.entity.artifact_entity import RecommendationTableArtifact
from recsys.entity.config_entity import RecommendationTableConfig
from recsys.exception import RecsysException
from recsys.logger import logger
from recsys.utils.common import save_json, sha256_of
from recsys.utils.shared import customer_id_to_uint64
from recsys.utils.tracking import git_commit


# ---------------------------------------------------------------- the serving format (plain functions, tested)

def check_list(articles, k, owner):
    """Every served list must be k different, real (non-zero) articles."""
    if len(articles) != k:
        raise ValueError(f"{owner} has {len(articles)} articles, expected {k}")
    if len(set(articles)) != k:
        raise ValueError(f"{owner} has a repeated article: articles must be distinct")
    if 0 in articles:
        raise ValueError(f"{owner} has article 0 (not a real article)")


def build_recs(recommendations, k):
    """Long table -> (recs, rec_row_of).
    recs:       one row per customer with history; row 0 stays all zeros (= "no history").
    rec_row_of: {customer_join_idx: row number in recs}."""
    # Step 1: one list per customer, in rank order
    recommendations = recommendations.sort_values(['customer_join_idx', 'rank'])
    lists = recommendations.groupby('customer_join_idx')['article_id'].apply(list)

    # Step 2: an empty array with one extra row (row 0), then fill it customer by customer
    recs = np.zeros((len(lists) + 1, k), dtype=np.int32)
    rec_row_of = {}
    row = 1
    for customer, articles in lists.items():
        check_list(articles, k, f"customer {customer}")
        recs[row] = articles
        rec_row_of[customer] = row
        row = row + 1
    return recs, rec_row_of


def build_bucket_lists(bucket_table, k):
    """Long table -> array with one row per age bucket (row b = bucket b; row 0 = the global list)."""
    bucket_table = bucket_table.sort_values(['age_bucket', 'rank'])
    lists = bucket_table.groupby('age_bucket')['article_id'].apply(list)

    bucket_lists = np.zeros((len(lists), k), dtype=np.int32)
    for bucket, articles in lists.items():
        check_list(articles, k, f"bucket {bucket}")
        bucket_lists[bucket] = articles            # an IndexError here would mean a bucket number is missing
    return bucket_lists


def build_lookup(lookup, rec_row_of):
    """customer_lookup -> three arrays in the SAME order, sorted by 64-bit id:
    ids (uint64), rec_rows (row in recs, 0 = no history), buckets (stored age bucket)."""
    table = pd.DataFrame({
        'id': np.array([customer_id_to_uint64(customer_id) for customer_id in lookup['customer_id']], dtype=np.uint64),
        'rec_row': [rec_row_of.get(customer, 0) for customer in lookup['customer_join_idx']],
        'bucket': lookup['age_bucket'],
    })

    # Sorted so the API can use np.searchsorted (binary search: about 20 steps for 1.4M customers)
    table = table.sort_values('id')

    # Two customers with the same 64-bit id would be mixed up: refuse instead
    if table['id'].duplicated().any():
        raise ValueError("64-bit customer ids are not unique: two customers would share one id")
    if (table['rec_row'] > 0).sum() != len(rec_row_of):
        raise ValueError("Some customers with a list are missing from the lookup")

    ids = table['id'].to_numpy(dtype=np.uint64)
    rec_rows = table['rec_row'].to_numpy(dtype=np.int32)
    buckets = table['bucket'].to_numpy(dtype=np.int8)
    return ids, rec_rows, buckets


def table_version(file_checksums):
    """ONE checksum for the whole build: made from every file's checksum, so it changes if ANY file changes."""
    text = ""
    for name in sorted(file_checksums):
        text = text + f"{name}:{file_checksums[name]}\n"
    return hashlib.sha256(text.strip().encode()).hexdigest()


def write_serving_files(recommendations, bucket_table, lookup, k, history_end_week, age_edges, folder):
    """The serving format + its manifest. Returns (manifest path, table_version)."""
    recs, rec_row_of = build_recs(recommendations, k)
    bucket_lists = build_bucket_lists(bucket_table, k)
    ids, rec_rows, buckets = build_lookup(lookup, rec_row_of)

    # Save the five arrays, then fingerprint each saved file
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {
        'recs.npy': recs,
        'bucket_lists.npy': bucket_lists,
        'lookup_ids.npy': ids,
        'lookup_rec_row.npy': rec_rows,
        'lookup_bucket.npy': buckets,
    }
    checksums = {}
    shapes = {}
    for name, array in arrays.items():
        np.save(folder / name, array)
        checksums[name] = sha256_of(folder / name)
        shapes[name] = list(array.shape)

    # The manifest: what the API checks at startup before it serves anything
    commit, dirty = git_commit()
    manifest = {
        'table_version': table_version(checksums),
        'sha256': checksums,
        'git_commit': commit,
        'git_dirty': dirty,
        'history_end_week': history_end_week,
        'age_edges': list(age_edges),        # the API buckets ages with THESE edges, never its own copy
        'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'shapes': shapes,
    }
    manifest_path = folder / 'serving_manifest.json'
    save_json(manifest, manifest_path)

    megabytes = (recs.nbytes + bucket_lists.nbytes + ids.nbytes + rec_rows.nbytes + buckets.nbytes) / 1024 ** 2
    logger.info(f"Serving files: recs {recs.shape}, buckets {bucket_lists.shape}, lookup {len(ids):,} ids, "
                f"{megabytes:.1f} MB | table_version {manifest['table_version'][:12]}")
    return manifest_path, manifest['table_version']


# ---------------------------------------------------------------- the stage

class RecommendationTable:
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

            rows = []
            for customer in warm_ids:
                for rank, article in enumerate(lists[customer], start=1):
                    rows.append((customer, rank, article))
            recommendations = pd.DataFrame(rows, columns=['customer_join_idx', 'rank', 'article_id']).astype('int32')

            # 2. The age-bucket lists (already filled to k by ListBuilder; bucket 0 = global)
            bucket_lists = pd.read_parquet(c.age_bucket_path)
            if len(bucket_lists) != len(c.age_edges) * c.k:
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

            # 4. Format 1: long Parquet (for checking)
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

            # 5. Format 2: the serving arrays (for answering requests); build_recs re-checks every list
            manifest_path, version = write_serving_files(recommendations, bucket_lists, lookup, c.k,
                                                         c.history_end_week, c.age_edges, c.output_dir / 'serving')

            logger.info(f"Recommendation table finished in {time.perf_counter() - start_time:.1f}s -> {c.output_dir}")
            return RecommendationTableArtifact(
                recommendations_path=recommendations_path,
                bucket_lists_path=bucket_lists_path,
                customer_lookup_path=customer_lookup_path,
                serving_manifest_path=manifest_path,
                table_version=version,
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
