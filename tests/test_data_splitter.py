import pandas as pd

from recsys.components.data_splitter import DataSplitter
from recsys.entity.config_entity import DataSplitterConfig

START = pd.Timestamp('2020-07-01')       # a Wednesday: week 0 starts here


def test_split_boundaries(tmp_path):
    # 09-08 = 69 days after 07-01 -> week 9 (last train day, a Tuesday)
    # 09-09 = 70 days -> week 10 (first validation day, a Wednesday); 09-16 = 77 days -> week 11 (test)
    transactions = pd.DataFrame({
        't_dat': pd.to_datetime(['2020-09-08', '2020-09-09', '2020-09-16']),
        'customer_join_idx': [1, 2, 3],
        'article_id': [100, 200, 300],
        'price': [0.01, 0.02, 0.03],
        'sales_channel_id': [1, 2, 1],
    })
    transactions['week'] = (transactions['t_dat'] - START).dt.days // 7       # the same formula as ingestion
    transactions.to_parquet(tmp_path / 'transactions.parquet', index=False)

    config = DataSplitterConfig(
        transactions_path=tmp_path / 'transactions.parquet',
        train_first_week=0, train_last_week=9, val_week=10, test_week=11,
        expected_train=None, expected_val=None, expected_test=None,          # tiny data: no count check
        output_dir=tmp_path / 'splits',
    )
    artifact = DataSplitter(config).run()

    train = pd.read_parquet(artifact.train_path)
    val = pd.read_parquet(artifact.val_path)
    assert train['t_dat'].tolist() == [pd.Timestamp('2020-09-08')]          # last day of week 9 -> train
    assert val['t_dat'].tolist() == [pd.Timestamp('2020-09-09')]            # first day of week 10 -> validation
