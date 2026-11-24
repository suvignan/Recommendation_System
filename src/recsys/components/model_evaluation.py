import time

import numpy as np
import pandas as pd

from recsys.components.list_builder import age_to_bucket, fill_up
from recsys.entity.artifact_entity import ModelEvaluationArtifact
from recsys.entity.config_entity import ModelEvaluationConfig
from recsys.exception import RecsysException
from recsys.logger import logger
from recsys.utils.common import save_json
from recsys.utils.metrics import score


def long_to_lists(table, key):
    """Long format (key, rank, article_id) -> {key: [articles in rank order]}."""
    table = table.sort_values([key, 'rank'])
    return table.groupby(key)['article_id'].apply(list).to_dict()


def load_map(path):
    """The trainer's parquet mapping (value, index) -> {id: row/column number}."""
    table = pd.read_parquet(path)
    return dict(zip(table['value'], table['index']))


def column_to_article_array(article_map):
    """ALS column number -> article ID (column 0 = unknown). array[column] gives the article."""
    array = np.zeros(len(article_map) + 1, dtype='int64')
    for article, column in article_map.items():
        array[column] = article
    return array


def als_recommend(customer_ids, k, user_factors, item_factors, customer_map, column_to_article):
    """ALS top-k articles per customer, best first: score = customer vector . every article vector (notebook 08)."""
    rows = np.array([customer_map[c] for c in customer_ids])
    top_columns = []
    for start in range(0, len(rows), 2000):                                   # 2,000 customers at a time: memory
        chunk_scores = user_factors[rows[start:start + 2000]] @ item_factors.T
        chunk_scores[:, 0] = -np.inf          # column 0 = 'unknown', no real article: never recommend it (Day 15)
        top = np.argpartition(-chunk_scores, k, axis=1)[:, :k]                # the k best, unordered (fast)
        order = np.take_along_axis(chunk_scores, top, axis=1).argsort(axis=1)[:, ::-1]   # sort only those k
        top_columns.append(np.take_along_axis(top, order, axis=1))
    return column_to_article[np.vstack(top_columns)]


def warm_lists(warm_ids, own_lists, als_parts, k, filler_candidates):
    """The frozen warm path: own purchases first, then ALS top filler_candidates fills to k.
    ONE function for evaluation (week 9) and the serving table (week 11)."""
    lists = {}
    for customer, candidates in zip(warm_ids, als_recommend(warm_ids, filler_candidates, *als_parts)):
        lists[customer] = fill_up(own_lists[customer], candidates, k)
    return lists


def load_als_parts(user_factors_path, item_factors_path, customer_map_path, article_map_path):
    """The trainer's saved files -> (user vectors, item vectors, customer map, column -> article array)."""
    return (np.load(user_factors_path), np.load(item_factors_path),
            load_map(customer_map_path), column_to_article_array(load_map(article_map_path)))


def summarise(recall, ndcg, is_cold):
    """Average recall and NDCG for all / warm / cold customers."""
    parts = {'all': np.ones(len(is_cold), dtype=bool), 'warm': ~is_cold, 'cold': is_cold}
    return {name: {'recall': float(recall[mask].mean()), 'ndcg': float(ndcg[mask].mean())}
            for name, mask in parts.items()}


def gate_checks(final, popularity, bad_lists, reference_recall, reference_tolerance,
                min_lift_over_popularity, cold_must_beat_popularity):
    """The params.yaml gate, one entry per rule: {'check', 'passed', 'detail'}. Collects ALL results (doesn't stop at the first)."""
    checks = []

    def record(name, passed, detail):
        checks.append({'check': name, 'passed': bool(passed), 'detail': detail})

    record('sanity: every list = k distinct articles, none missing', bad_lists == 0, f"{bad_lists} bad lists")

    if reference_recall is None:
        record('snapshot parity', True, 'skipped: reference_recall is null (live run)')
    else:
        difference = abs(final['all']['recall'] - reference_recall)
        record('snapshot parity', difference <= reference_tolerance,
               f"recall {final['all']['recall']} vs reference {reference_recall}: difference {difference:.2e} "
               f"(tolerance {reference_tolerance:.0e})")

    # On tiny data (the smoke-test fixture) popularity can score exactly 0: no division by zero
    lift = final['all']['recall'] / popularity['all']['recall'] if popularity['all']['recall'] > 0 else float('inf')
    record('live gate: lift over popularity', lift >= min_lift_over_popularity,
           f"{final['all']['recall']:.4f} / {popularity['all']['recall']:.4f} = {lift:.2f}x (need >= {min_lift_over_popularity}x)")

    if cold_must_beat_popularity:
        record('live gate: cold beats popularity', final['cold']['recall'] >= popularity['cold']['recall'],
               f"cold {final['cold']['recall']:.4f} vs popularity cold {popularity['cold']['recall']:.4f}")
    return checks


class ModelEvaluation:
    """Rebuilds the frozen system's top k for every validation customer FROM THE SAVED FILES, scores it, writes a report."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def run(self) -> ModelEvaluationArtifact:
        try:
            c = self.config
            start_time = time.perf_counter()

            # The answer key: one row per validation customer, with the set of articles they bought
            val = pd.read_parquet(c.val_path, columns=['customer_join_idx', 'article_id', 'week'])
            if set(val['week'].unique()) != {c.history_end_week + 1}:
                raise ValueError(f"Validation weeks {sorted(val['week'].unique())} are not the week after "
                                 f"history_end_week {c.history_end_week}: lists and answer key don't line up")
            answer_key = val.groupby('customer_join_idx')['article_id'].unique()
            customer_ids = answer_key.index.to_list()

            # The saved inputs of the frozen system
            own_lists = long_to_lists(pd.read_parquet(c.repurchase_path), 'customer_join_idx')
            bucket_lists = long_to_lists(pd.read_parquet(c.age_bucket_path), 'age_bucket')
            als_parts = load_als_parts(c.user_factors_path, c.item_factors_path, c.customer_map_path, c.article_map_path)

            # Warm = bought something in the history = has a repurchase list
            warm_ids = [customer for customer in customer_ids if customer in own_lists]
            cold_ids = [customer for customer in customer_ids if customer not in own_lists]

            recommendations = warm_lists(warm_ids, own_lists, als_parts, c.k, c.filler_candidates)

            ages = pd.read_parquet(c.customers_path, columns=['customer_join_idx', 'age']).set_index('customer_join_idx')['age']
            cold_buckets = age_to_bucket(ages.loc[cold_ids], c.age_edges)
            for customer in cold_ids:
                recommendations[customer] = bucket_lists[cold_buckets[customer]]   # already filled to k; 0 = global

            lists = [recommendations[customer] for customer in customer_ids]
            bad_lists = sum(1 for recs in lists
                            if len(recs) != c.k or len(set(recs)) != c.k or pd.isna(pd.Series(recs)).any())

            is_cold = np.array([customer not in own_lists for customer in customer_ids])
            recall, ndcg = score(lists, answer_key.to_list(), c.k)
            final = summarise(recall, ndcg, is_cold)

            # The baseline, measured in the SAME run on the SAME customers: last-week top k for everyone
            last_week = pd.read_parquet(c.last_week_path).sort_values('rank')['article_id'].to_list()
            popularity_recall, popularity_ndcg = score([last_week[:c.k]] * len(customer_ids), answer_key.to_list(), c.k)
            popularity = summarise(popularity_recall, popularity_ndcg, is_cold)

            checks = gate_checks(final, popularity, bad_lists, c.reference_recall, c.reference_tolerance,
                                 c.min_lift_over_popularity, c.cold_must_beat_popularity)
            gate_passed = all(check['passed'] for check in checks)

            # WARNING, not a gate ("worth a look", never stops the run): is the ALS filler still earning its place?
            # Warm customers, own purchases filled from last-week popularity instead of ALS. ALS's gain was 1.08x on
            # validation but only 1.01x on test, so a hard rule here would fail on normal weeks and get switched off.
            warm_rows = ~is_cold
            popularity_fill = [fill_up(own_lists[customer], last_week, c.k) for customer in warm_ids]
            popularity_fill_recall, _ = score(popularity_fill, answer_key[warm_rows].to_list(), c.k)
            baseline = popularity_fill_recall.mean()
            filler_gain = final['warm']['recall'] / baseline if baseline > 0 else float('inf')
            warnings = [{
                'check': 'ALS filler beats popularity filler (warm customers)',
                'flagged': bool(filler_gain < 1.0),
                'detail': f"warm recall {final['warm']['recall']:.4f} vs repurchase + popularity fill "
                          f"{popularity_fill_recall.mean():.4f} = {filler_gain:.3f}x (flag below 1.0x)",
            }]

            report = {
                'history_end_week': c.history_end_week,
                'scored_week': c.history_end_week + 1,
                'k': c.k,
                'customers': {'all': len(customer_ids), 'warm': len(warm_ids), 'cold': len(cold_ids)},
                'final_system': final,
                'last_week_popularity': popularity,
                'gate': {'passed': gate_passed, 'checks': checks},
                'warnings': warnings,
            }
            save_json(report, c.report_path)                      # the report FIRST, so a failed run still explains itself

            logger.info(f"Customers: {len(customer_ids):,} (warm {len(warm_ids):,}, cold {len(cold_ids):,})")
            for part in ('all', 'warm', 'cold'):
                logger.info(f"Final system {part:4}: recall@{c.k} {final[part]['recall']:.4f}  NDCG@{c.k} {final[part]['ndcg']:.4f}"
                            f"  | popularity recall {popularity[part]['recall']:.4f}")
            for check in checks:
                logger.info(f"Gate {'PASS' if check['passed'] else 'FAIL'} | {check['check']}: {check['detail']}")
            for warning in warnings:
                log = logger.warning if warning['flagged'] else logger.info
                log(f"Warning {'FLAGGED' if warning['flagged'] else 'ok'} | {warning['check']}: {warning['detail']}")

            if not gate_passed:
                failed = [check['check'] for check in checks if not check['passed']]
                raise ValueError(f"Model evaluation gate failed: {failed} (see {c.report_path})")

            logger.info(f"Model evaluation finished in {time.perf_counter() - start_time:.1f}s -> {c.report_path}")
            return ModelEvaluationArtifact(
                report_path=c.report_path,
                recall_all=final['all']['recall'],
                ndcg_all=final['all']['ndcg'],
                gate_passed=gate_passed,
            )

        except Exception as error:
            raise RecsysException(error) from error
