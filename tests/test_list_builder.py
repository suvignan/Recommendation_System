import pandas as pd

from recsys.components.list_builder import (
    ListBuilder,
    repurchase_lists,
    age_to_bucket,
    last_week_top,
)
from recsys.entity.config_entity import ListBuilderConfig

START = pd.Timestamp('2020-07-01')

EDGES = (16, 20, 25, 30, 40, 50, 60, 100)


def make_history(rows):
    """Create a history DataFrame with a calculated week column."""

    history = pd.DataFrame(
        rows,
        columns=['t_dat', 'customer_join_idx', 'article_id']
    )

    history['t_dat'] = pd.to_datetime(history['t_dat'])

    history['week'] = (
        (history['t_dat'] - START).dt.days // 7
    )

    return history

def test_repurchase_newest_first():
    history = make_history([('2020-07-01',1,100),('2020-07-15',1,200)])
    lists = repurchase_lists(history, 10)
    assert lists[1] == [200, 100]

def test_repurchase_no_duplicates():
    history = make_history([
        ('2020-07-01', 1, 100),
        ('2020-07-03', 1, 100)
    ])

    lists = repurchase_lists(history, 10)

    assert lists[1] == [100]

def test_repurchase_max_limit():
    rows = [
        (f'2020-07-{day:02d}', 1, day)
        for day in range(1,13)
    ]
    history = make_history(rows)
    lists = repurchase_lists(history, 10)
    assert len(lists[1]) ==10
    assert lists[1] == [12,11,10,9,8,7,6,5,4,3]

def test_cold_customer_has_no_repurchase_list():
    all_rows = make_history([
        ('2020-07-01', 1, 100),
        ('2020-07-15', 9, 200)
    ])

    history = all_rows[all_rows['week'] <= 1]

    lists = repurchase_lists(history, 10)

    assert 1 in lists
    assert 9 not in lists


# ---- Tests 5-6 run the whole ListBuilder class on tiny files in tmp_path, with a config built BY HAND
# (no YAML, no real data: the payoff of passing a config into every component)

def run_list_builder(tmp_path, history, customers, history_end_week, k):
    history.to_parquet(tmp_path / 'transactions.parquet', index=False)
    customers.to_parquet(tmp_path / 'customers.parquet', index=False)
    config = ListBuilderConfig(
        transactions_path=tmp_path / 'transactions.parquet',
        customers_path=tmp_path / 'customers.parquet',
        history_end_week=history_end_week,
        k=k,                                   # small k = small lists you can check by eye
        repurchase_max=10,
        filler_candidates=k,
        age_edges=EDGES,
        output_dir=tmp_path / 'lists',
    )
    artifact = ListBuilder(config).run()
    return pd.read_parquet(artifact.age_bucket_path), pd.read_parquet(artifact.last_week_path)


def week2_story():
    """Week 2 (07-15): two young buyers (bucket 1), one with no age, seven 55-year-olds (bucket 6).
    Counts: article 1 x7, 2 x6, 3 x5, 4 x4, 5 x3, 901 x2, 902 x1 -> no ties, so every expected answer is certain."""
    customers = pd.DataFrame({
        'customer_join_idx': [1, 2, 3, 10, 11, 12, 13, 14, 15, 16],
        'age': [18.0, 19.0, None, 55, 55, 55, 55, 55, 55, 55],
    })
    rows = [('2020-07-15', 1, 901), ('2020-07-15', 2, 901), ('2020-07-15', 2, 902)]
    for article, times in [(1, 7), (2, 6), (3, 5), (4, 4), (5, 3)]:
        for customer in range(10, 10 + times):
            rows.append(('2020-07-15', customer, article))
    return make_history(rows), customers


def bucket_list(age_table, bucket):
    return age_table[age_table['age_bucket'] == bucket].sort_values('rank')['article_id'].tolist()


def test_missing_age_gets_global_list(tmp_path):
    history, customers = week2_story()
    age_table, last_week = run_list_builder(tmp_path, history, customers, history_end_week=2, k=5)

    assert age_to_bucket(pd.Series([None]), EDGES).tolist() == [0]        # no age -> bucket 0
    assert bucket_list(age_table, 0) == [1, 2, 3, 4, 5]                    # bucket 0 = the global top 5
    assert last_week.sort_values('rank')['article_id'].tolist() == [1, 2, 3, 4, 5]


def test_short_bucket_filled_from_global_without_duplicates(tmp_path):
    history, customers = week2_story()
    age_table, _ = run_list_builder(tmp_path, history, customers, history_end_week=2, k=5)

    young = bucket_list(age_table, 1)          # own list [901, 902] is shorter than k=5
    assert young == [901, 902, 1, 2, 3]        # own articles first, then the global list in order
    assert len(young) == 5
    assert len(set(young)) == len(young)       # no article twice


def test_same_date_smaller_article_first():
    history = make_history([('2020-07-01', 1, 500), ('2020-07-01', 1, 300)])
    assert repurchase_lists(history, 10)[1] == [300, 500]


def test_same_count_smaller_article_first():
    history = make_history([('2020-07-01', 1, 500), ('2020-07-01', 2, 300), ('2020-07-01', 3, 700), ('2020-07-01', 4, 700)])
    assert last_week_top(history, 0, 3) == [700, 300, 500]       # 700 bought twice; 300 and 500 tie on once
