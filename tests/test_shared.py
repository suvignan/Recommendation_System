"""The API's plain-Python age_bucket_for must give EXACTLY the pipeline's age_to_bucket (pandas) for every age.
Two implementations of one rule are only safe if a test proves they agree."""
import pandas as pd

from recsys.components.list_builder import age_to_bucket
from recsys.utils.shared import age_bucket_for

EDGES = [16, 20, 25, 30, 40, 50, 60, 100]


def test_api_age_bucket_equals_pipeline_age_bucket():
    ages = [age / 2 for age in range(0, 241)] + [None, 15.999, 16.0, 99.999, 100.0, 150.0]   # 0, 0.5, ... 120
    pipeline = age_to_bucket(pd.Series(ages, dtype="float64"), EDGES).tolist()
    api = [age_bucket_for(age, EDGES) for age in ages]
    assert api == pipeline


def test_examples():
    assert age_bucket_for(25, EDGES) == 3          # 25 <= age < 30
    assert age_bucket_for(7, EDGES) == 0           # under 16 -> no bucket
    assert age_bucket_for(None, EDGES) == 0
