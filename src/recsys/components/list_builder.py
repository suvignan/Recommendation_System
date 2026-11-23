import time
import pandas as pd
from recsys.components.data_ingestion import DataIngestion
from recsys.entity.artifact_entity import ListBuilderArtifact
from recsys.entity.config_entity import ListBuilderConfig, get_ingestion_config, get_list_builder_config
from recsys.exception import RecsysException
from recsys.logger import logger

# The non-model inputs of the frozen system, moved from notebook 08 Step 4b.
# Plain functions (not methods), so tests can call them directly on tiny hand-made tables.
# Day 14 Step 0, TIE RULE (chosen before measuring, arbitrary on purpose so it can't be tuning):
#   same date -> smaller article_id first;  same count -> smaller article_id first.
#   Before this, ties were ordered by row position in the file: new data in a new order could change the lists.
# This file is the ONLY copy: notebooks 07, 08 and 09 import these functions.


def top_articles(article_ids, k):
    """The k most bought articles, most first; equal counts -> smaller article_id first."""
    counts = article_ids.value_counts().rename_axis('article_id').reset_index(name='count')
    counts = counts.sort_values(['count', 'article_id'], ascending=[False, True])
    return counts['article_id'].head(k).to_list()


def repurchase_lists(history, max_items):
    """{customer: [own articles, newest first, each once, at most max_items]}; same date -> smaller article_id first"""
    sorted_history = history.sort_values(
        ['t_dat', 'article_id'],
        ascending=[False, True]
    )
    unique_articles = sorted_history.drop_duplicates(          # newest copy of each article survives
        subset=['customer_join_idx', 'article_id'],
        keep='first'
    )
    limited_articles = unique_articles.groupby(
        'customer_join_idx'
    ).head(max_items)
    result = limited_articles.groupby('customer_join_idx')['article_id'].apply(list).to_dict()

    return result


def last_week_top(history, week, k):
    """The k best sellers of one week, most bought first."""
    return top_articles(history[history['week'] == week]['article_id'], k)


def age_to_bucket(ages, edges):
    """Ages -> buckets 1..len(edges)-1 like the Day 5 vocabulary; missing or out of range -> 0 (unknown)."""
    # Found by test 5: a column of only None has dtype object and pd.cut crashes on it. Numbers first, anything else -> NaN -> 0
    ages = pd.to_numeric(ages, errors='coerce')
    buckets = pd.cut(
        ages,
        bins=list(edges),
        right=False, labels=False
    )

    return (buckets + 1).fillna(0).astype(int)


def age_bucket_top10(history, customers, week, edges, k):
    """{bucket: top k articles bought that week by buyers in that bucket}; bucket 0 (unknown age) is left out."""
    week_rows = history[history['week'] == week][['customer_join_idx', 'article_id']]
    customers = customers.copy()
    customers['age_bucket'] = age_to_bucket(customers['age'], edges)
    week_rows = week_rows.merge(customers[['customer_join_idx', 'age_bucket']],
                                on='customer_join_idx', how='left', validate='many_to_one')

    lists = {}
    for bucket in sorted(week_rows["age_bucket"].unique()):
        if bucket == 0:
            continue
        lists[int(bucket)] = top_articles(week_rows[week_rows["age_bucket"] == bucket]['article_id'], k)
    return lists


def fill_up(my_list, fallback, k=10):
    """Add fallback articles (skipping ones already in my_list) until the list has k articles."""
    filled = list(my_list)
    for article in fallback:
        if len(filled) == k:
            break
        if article not in filled:
            filled.append(article)
    return filled


def to_long(lists, key_name):
    """{key: [articles]} -> one row per (key, rank, article); rank 1 = first."""
    rows = []
    for key, articles in lists.items():
        for rank, article in enumerate(articles, start=1):
            rows.append((key, rank, article))
    return pd.DataFrame(rows, columns=[key_name, 'rank', 'article_id'])


class ListBuilder:
    """Builds the lists for ONE moment: history = weeks 0..history_end_week (9 = evaluation, 11 = serving)."""

    def __init__(self, config: ListBuilderConfig):
        self.config = config

    def run(self) -> ListBuilderArtifact:
        try:
            start_time = time.perf_counter()
            c = self.config

            transactions = pd.read_parquet(c.transactions_path)
            history = transactions[transactions['week'] <= c.history_end_week]          # THE setting
            customers = pd.read_parquet(c.customers_path, columns=['customer_join_idx', 'age'])
            logger.info(f"History: weeks 0-{c.history_end_week}, {len(history):,} rows")

            own = repurchase_lists(history, c.repurchase_max)
            last_week = last_week_top(history, c.history_end_week, c.filler_candidates)    # top 30
            global_top = last_week[:c.k]                                                   # top 10
            buckets = age_bucket_top10(history, customers, c.history_end_week, c.age_edges, c.k)

            # Fill every bucket to k NOW, so the API only ever looks up (bucket 0 = unknown age = the global list)
            served_buckets = {0: global_top}
            for bucket in range(1, len(c.age_edges)):
                served_buckets[bucket] = fill_up(buckets.get(bucket, []), global_top, c.k)

            repurchase = to_long(own, 'customer_join_idx')
            age_table = to_long(served_buckets, 'age_bucket')
            last_week_table = pd.DataFrame({'rank': range(1, len(last_week) + 1), 'article_id': last_week})
            for table in (repurchase, age_table, last_week_table):
                for column in ('customer_join_idx', 'age_bucket', 'rank', 'article_id'):
                    if column in table:
                        table[column] = table[column].astype('int32')

            c.output_dir.mkdir(parents=True, exist_ok=True)
            repurchase_path = c.output_dir / 'repurchase.parquet'
            last_week_path = c.output_dir / 'last_week.parquet'
            age_bucket_path = c.output_dir / 'age_buckets.parquet'
            repurchase.to_parquet(repurchase_path, index=False)
            last_week_table.to_parquet(last_week_path, index=False)
            age_table.to_parquet(age_bucket_path, index=False)

            logger.info(f"Repurchase lists: {len(own):,} customers, {len(repurchase):,} rows "
                        f"({len(repurchase) / len(own):.2f} per customer)")
            logger.info(f"Last-week list: {len(last_week_table)} rows | age-bucket lists: {len(age_table)} rows")
            logger.info(f"List builder finished in {time.perf_counter() - start_time:.1f}s -> {c.output_dir}")

            return ListBuilderArtifact(
                repurchase_path=repurchase_path,
                last_week_path=last_week_path,
                age_bucket_path=age_bucket_path,
                history_end_week=c.history_end_week,
                n_repurchase_customers=len(own),
            )

        except Exception as error:
            raise RecsysException(error) from error


if __name__ == "__main__":
    ingestion_artifact = DataIngestion(get_ingestion_config()).run()
    print(ListBuilder(get_list_builder_config(ingestion_artifact, history_end_week=9)).run())
