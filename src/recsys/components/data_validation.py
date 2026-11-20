import time
import pandas as pd
from recsys.components.data_ingestion import DataIngestion
from recsys.entity.config_entity import DataValidationConfig, get_ingestion_config, get_validation_config
from recsys.entity.artifact_entity import DataValidationArtifact
from recsys.utils.common import save_json
from recsys.logger import logger
from recsys.exception import RecsysException

# The schema stores Parquet type names for the raw files; pandas calls the same types differently
PARQUET_TO_PANDAS = {'string': 'str', 'double': 'float64', 'int64': 'int64'}


class DataValidation:
    """Is ingestion's output safe to build a model on? Runs EVERY check, writes the report, THEN stops if any failed."""

    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.results = []

    def record(self, name, passed, details):
        # Record, don't crash: every check runs even if an earlier one failed
        self.results.append({
            'check': name,
            'passed': bool(passed),        # numpy's bool_ can't go into JSON
            'details': details
        })

    def read_data(self):

        transactions = pd.read_parquet(
            self.config.transactions_path
        )

        customers = pd.read_parquet(
            self.config.customers_path
        )

        articles = pd.read_parquet(
            self.config.articles_path
        )

        return transactions, customers, articles

    def compare_columns(self, name, actual_dtypes, expected):
        # Shared by the three schema checks: (column -> dtype) the table HAS vs the contract PROMISES
        actual = {
            column: str(dtype)
            for column, dtype in actual_dtypes.items()
        }

        passed = actual == expected

        differences = {
            column: (actual.get(column), expected.get(column))      # (found, expected)
            for column in set(actual) | set(expected)
            if actual.get(column) != expected.get(column)
        }

        self.record(
            name,
            passed,
            f'differences (found, expected): {differences}'
        )

    # Check 1: transactions (the contract is already in pandas names)
    def validate_transactions_schema(self, transactions):
        self.compare_columns('transactions schema', transactions.dtypes, dict(self.config.transaction_columns))

    # Check 2: customers (Parquet names -> pandas names; ingestion ADDS customer_join_idx)
    def validate_customers_schema(self, customers):
        expected = {
            column: PARQUET_TO_PANDAS[dtype]
            for column, dtype in self.config.customer_columns.items()
        }
        expected['customer_join_idx'] = 'int64'
        self.compare_columns('customers schema', customers.dtypes, expected)

    # Check 3: articles
    def validate_articles_schema(self, articles):
        expected = {
            column: PARQUET_TO_PANDAS[dtype]
            for column, dtype in self.config.article_columns.items()
        }
        self.compare_columns('articles schema', articles.dtypes, expected)

    # Check 4
    def validate_unique_ids(self, customers, articles):

        dup_customers = customers['customer_id'].duplicated().sum()

        dup_articles = articles['article_id'].duplicated().sum()

        passed = dup_customers == 0 and dup_articles == 0

        self.record(
            'unique IDs',
            passed,
            f'customer duplicates: {dup_customers}, '
            f'article duplicates: {dup_articles}'
        )

    # Check 5 (age, club status etc. are ALLOWED to be missing, Day 5)
    def validate_missing_values(self, transactions, customers):

        missing = {
            't_dat': int(transactions['t_dat'].isna().sum()),
            'customer_join_idx': int(transactions['customer_join_idx'].isna().sum()),
            'article_id': int(transactions['article_id'].isna().sum()),
            'customer_id': int(customers['customer_id'].isna().sum())
        }

        passed = sum(missing.values()) == 0

        self.record(
            'missing values in key columns',
            passed,
            f'missing counts: {missing}'
        )

    # Check 6 (Day 3)
    def validate_dates(self, transactions):

        first = transactions['t_dat'].min()
        last = transactions['t_dat'].max()

        start = pd.Timestamp(self.config.window_start)
        expected_last = start + pd.Timedelta(days=7 * self.config.window_weeks - 1)

        first_valid = first == start and first.day_name() == 'Wednesday'
        last_valid = last == expected_last

        dates_per_week = transactions.groupby('week')['t_dat'].nunique()
        bad_weeks = dates_per_week[dates_per_week != 7]
        weeks_valid = len(bad_weeks) == 0 and len(dates_per_week) == self.config.window_weeks

        passed = first_valid and last_valid and weeks_valid

        self.record(
            'dates',
            passed,
            (
                f'first: {first.date()}, expected: {start.date()}, '
                f'first is Wednesday: {first.day_name() == "Wednesday"}; '
                f'last: {last.date()}, expected: {expected_last.date()}; '
                f'weeks: {len(dates_per_week)} (expected {self.config.window_weeks}); '
                f'weeks with != 7 dates: {bad_weeks.to_dict()}'
            )
        )

    # Check 7: every purchase points to a real article and a real customer
    def validate_references(self, transactions, customers, articles):
        unknown_articles = int((~transactions['article_id'].isin(articles['article_id'])).sum())
        unknown_customers = int((~transactions['customer_join_idx'].isin(customers['customer_join_idx'])).sum())
        self.record(
            'transactions reference real articles and customers',
            unknown_articles == 0 and unknown_customers == 0,
            f'unknown articles: {unknown_articles}, unknown customers: {unknown_customers}'
        )

    # Check 8: category values are in the schema's allowed list (after the alias); missing is allowed
    def validate_allowed_values(self, customers):
        for column, allowed in self.config.allowed_values.items():
            present = set(customers[column].dropna())
            unknown = sorted(present - set(allowed))
            self.record(
                f'allowed values: {column}',
                len(unknown) == 0,
                f'unknown values: {unknown}'
            )

    # Check 9: ingestion's dedup really happened
    def validate_no_duplicates(self, transactions):
        remaining = int(transactions.duplicated(subset=list(self.config.dedup_key)).sum())
        self.record(
            'no same-day duplicates',
            remaining == 0,
            f'duplicates on {list(self.config.dedup_key)}: {remaining}'
        )

    # Check 10: only when an expected count is set (null in params = live weekly run)
    def validate_row_count(self, transactions):
        expected = self.config.expected_after_dedup
        if expected is None:
            self.record('row count', True, 'skipped: no expected count (live weekly run)')
        else:
            self.record(
                'row count',
                len(transactions) == expected,
                f'expected {expected:,}, found {len(transactions):,}'
            )

    # Check 11: the window filter compares t_dat as TEXT, which only works for 'YYYY-MM-DD' (counted by ingestion)
    def validate_raw_date_format(self):
        bad = self.config.raw_bad_date_rows
        self.record(
            "raw t_dat written 'YYYY-MM-DD'",
            bad == 0,
            f"raw rows not 'YYYY-MM-DD': {bad:,} (they would be silently left out of the window)"
        )

    def run(self) -> DataValidationArtifact:
        try:
            start_time = time.perf_counter()
            self.results = []
            transactions, customers, articles = self.read_data()

            self.validate_transactions_schema(transactions)
            self.validate_customers_schema(customers)
            self.validate_articles_schema(articles)
            self.validate_unique_ids(customers, articles)
            self.validate_missing_values(transactions, customers)
            self.validate_dates(transactions)
            self.validate_references(transactions, customers, articles)
            self.validate_allowed_values(customers)
            self.validate_no_duplicates(transactions)
            self.validate_row_count(transactions)
            self.validate_raw_date_format()

            failed = [result['check'] for result in self.results if not result['passed']]
            report = {
                'validation_passed': len(failed) == 0,
                'checks_run': len(self.results),
                'failed': failed,
                'checks': self.results,
            }
            save_json(report, self.config.report_path)          # FIRST: the evidence survives the crash below

            logger.info(
                f"Data validation: {len(self.results) - len(failed)}/{len(self.results)} checks passed "
                f"in {time.perf_counter() - start_time:.1f}s -> {self.config.report_path}"
            )
            if failed:
                raise ValueError(f"Data validation failed: {failed}. Details in {self.config.report_path}")

            return DataValidationArtifact(report_path=self.config.report_path, validation_passed=True)

        except Exception as error:
            raise RecsysException(error) from error


if __name__ == "__main__":
    ingestion_artifact = DataIngestion(get_ingestion_config()).run()
    validation_artifact = DataValidation(get_validation_config(ingestion_artifact)).run()
    print(validation_artifact)
