import time
from datetime import timedelta

import pandas as pd

from recsys.entity.config_entity import (
    DataIngestionConfig,
    get_ingestion_config
)

from recsys.entity.artifact_entity import DataIngestionArtifact
from recsys.logger import logger
from recsys.exception import RecsysException


class DataIngestion:
    """Notebook 01 (load, number customers, 12-week window) + the start of notebook 03 (dedup, week), unchanged."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def run(self) -> DataIngestionArtifact:

        try:
            total_start = time.perf_counter()
            step_start = time.perf_counter()

            # 6a: Read transactions: only the window, only the needed columns
            # (t_dat is ISO text 'YYYY-MM-DD', and ISO text sorts like real dates, so the filter can compare strings)

            window_end = self.config.window_start + timedelta(
                weeks=self.config.window_weeks
            )

            transactions = pd.read_parquet(
                self.config.transactions_file,
                columns=list(self.config.transaction_columns),
                filters=[
                    ('t_dat', '>=', str(self.config.window_start)),
                    ('t_dat', '<', str(window_end))
                ]
            )

            rows_in_window = len(transactions)

            logger.info(
                f"Transactions loaded: {rows_in_window:,} rows from {self.config.window_start} "
                f"to before {window_end} in {time.perf_counter() - step_start:.1f}s"
            )

            # 6b: Read customers and articles

            step_start = time.perf_counter()
            customers = pd.read_parquet(
                self.config.customers_file,
                columns=list(self.config.customer_columns)
            )

            articles = pd.read_parquet(
                self.config.articles_file,
                columns=list(self.config.article_columns)
            )

            logger.info(f"Customers loaded: {len(customers):,}")
            logger.info(f"Articles loaded: {len(articles):,} in {time.perf_counter() - step_start:.1f}s")

            # Apply aliases (from the schema, so a new alias needs no code change)

            for column, mapping in self.config.aliases.items():
                customers[column] = customers[column].replace(mapping)

            logger.info("Customer aliases applied successfully")

            # 6c: Number the customers exactly as on Day 1: 0, 1, 2, ... in FILE ORDER (never sorted)

            step_start = time.perf_counter()
            customers['customer_join_idx'] = range(len(customers))

            customer_map = customers[['customer_id', 'customer_join_idx']]
            logger.info(f"Customer map created: {len(customer_map):,}")

            # 6d: Text ids -> numbers in transactions, then the Day 1 dtypes

            id_to_number = dict(zip(customer_map['customer_id'], customer_map['customer_join_idx']))
            transactions['customer_join_idx'] = transactions['customer_id'].map(id_to_number)

            missing_customers = transactions['customer_join_idx'].isna().sum()
            if missing_customers > 0:
                raise ValueError(
                    f'Found {missing_customers:,} transactions with missing customers'
                )
            transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
            transactions['customer_join_idx'] = transactions['customer_join_idx'].astype('int32')
            transactions['article_id'] = transactions['article_id'].astype('int32')
            transactions['price'] = transactions['price'].astype('float32')
            transactions['sales_channel_id'] = transactions['sales_channel_id'].astype('int8')

            # Same columns, same order as the notebook's table (the text id is no longer needed)
            transactions = transactions[['t_dat', 'customer_join_idx', 'article_id', 'price', 'sales_channel_id']]
            logger.info(f"Customer ids mapped, dtypes set in {time.perf_counter() - step_start:.1f}s")

            # 6e: Remove same-day duplicates (key from params; keep='first' in FILE order, so never sort before this)

            transactions = transactions.drop_duplicates(
                subset=list(self.config.dedup_key),
                keep='first'
            )
            rows_after_dedup = len(transactions)
            logger.info(
                f"Duplicates removed: {rows_in_window - rows_after_dedup:,} -> {rows_after_dedup:,} rows left"
            )

            # 6f: Week number, Day 3 formula: whole weeks since the window start (from params, not from the data)

            start = pd.Timestamp(self.config.window_start)
            transactions['week'] = (transactions['t_dat'] - start).dt.days // 7
            logger.info(f"Weeks: {transactions['week'].min()} to {transactions['week'].max()}")

            # 6g: Save the four tables and return the receipt

            output_dir = self.config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            transactions.to_parquet(output_dir / 'transactions.parquet', index=False)
            customers.to_parquet(output_dir / 'customers.parquet', index=False)
            articles.to_parquet(output_dir / 'articles.parquet', index=False)
            customer_map.to_parquet(output_dir / 'customer_map.parquet', index=False)

            logger.info(f"Data ingestion finished in {time.perf_counter() - total_start:.1f}s -> {output_dir}")

            return DataIngestionArtifact(
                transactions_path=output_dir / 'transactions.parquet',
                customers_path=output_dir / 'customers.parquet',
                articles_path=output_dir / 'articles.parquet',
                customer_map_path=output_dir / 'customer_map.parquet',
                rows_in_window=rows_in_window,
                rows_after_dedup=rows_after_dedup,
            )

        except Exception as error:
            raise RecsysException(error) from error


if __name__ == "__main__":
    artifact = DataIngestion(get_ingestion_config()).run()
    print(artifact)
