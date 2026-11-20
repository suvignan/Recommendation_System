import time

import pandas as pd

from recsys.components.data_ingestion import DataIngestion
from recsys.entity.artifact_entity import DataSplitterArtifact
from recsys.entity.config_entity import DataSplitterConfig, get_ingestion_config, get_splitter_config
from recsys.exception import RecsysException
from recsys.logger import logger


class DataSplitter:
    """Notebook 03: weeks 0-9 = train, week 10 = validation, week 11 = test, then prove nothing was lost or leaked."""

    def __init__(self, config: DataSplitterConfig):
        self.config = config

    def run(self) -> DataSplitterArtifact:
        try:
            start_time = time.perf_counter()
            c = self.config
            transactions = pd.read_parquet(c.transactions_path)

            # 2a: split by the week column (between() includes both ends)
            train = transactions[transactions['week'].between(c.train_first_week, c.train_last_week)]
            val = transactions[transactions['week'] == c.val_week]
            test = transactions[transactions['week'] == c.test_week]

            # 2b: the splitter's OWN logic: if one of these fails, the split is wrong, so stop immediately
            for name, frame, expected in [('train', train, c.expected_train),
                                          ('val', val, c.expected_val),
                                          ('test', test, c.expected_test)]:
                if expected is not None and len(frame) != expected:
                    raise ValueError(f'{name}: expected {expected:,} rows, got {len(frame):,}')

            if len(train) + len(val) + len(test) != len(transactions):       # conservation: no week lost or doubled
                raise ValueError(f'splits add up to {len(train) + len(val) + len(test):,}, '
                                 f'but ingestion has {len(transactions):,} rows')

            if not (train['t_dat'].max() < val['t_dat'].min() and val['t_dat'].max() < test['t_dat'].min()):
                raise ValueError('splits overlap in time: train must end before validation, validation before test')

            for name, frame in [('train', train), ('val', val), ('test', test)]:
                logger.info(f"{name}: {len(frame):,} rows, {frame['t_dat'].min().date()} to {frame['t_dat'].max().date()}")

            # 2c: save and return the receipt
            c.output_dir.mkdir(parents=True, exist_ok=True)
            paths = {}
            for name, frame in [('train', train), ('val', val), ('test', test)]:
                paths[name] = c.output_dir / f'{name}.parquet'
                frame.to_parquet(paths[name], index=False)

            logger.info(f"Data splitter finished in {time.perf_counter() - start_time:.1f}s -> {c.output_dir}")
            return DataSplitterArtifact(
                train_path=paths['train'], val_path=paths['val'], test_path=paths['test'],
                n_train=len(train), n_val=len(val), n_test=len(test),
            )

        except Exception as error:
            raise RecsysException(error) from error


if __name__ == "__main__":
    ingestion_artifact = DataIngestion(get_ingestion_config()).run()
    print(DataSplitter(get_splitter_config(ingestion_artifact)).run())
