"""Day 16 Step 1: the serving format (written by the table stage) on the hand-worked toy example (k = 2), plus its safety checks."""
import numpy as np
import pandas as pd
import pytest

from recsys.components.recommendation_table import build_bucket_lists, build_lookup, build_recs
from recsys.utils.shared import customer_id_to_uint64

ZEROS = "0" * 48       # pad to a 64-character id; only the last 16 hex characters matter


def toy():
    recommendations = pd.DataFrame({"customer_join_idx": [5, 5, 2, 2], "rank": [1, 2, 1, 2],
                                    "article_id": [900, 901, 800, 801]})
    lookup = pd.DataFrame({"customer_id": [ZEROS + "00000000000000ff", ZEROS + "0000000000000010",
                                           ZEROS + "0000000000000003"],
                           "customer_join_idx": [5, 2, 7], "age_bucket": [3, 0, 2]})
    return recommendations, lookup


def test_customer_id_to_uint64():
    assert customer_id_to_uint64(ZEROS + "00000000000000ff") == 255
    assert customer_id_to_uint64("f" * 64) == 2 ** 64 - 1               # the largest value still fits uint64


def test_build_recs_matches_the_toy():
    recs, rec_row_of = build_recs(toy()[0], k=2)
    assert recs.tolist() == [[0, 0], [800, 801], [900, 901]]           # row 0 unused; customer 2 -> 1, 5 -> 2
    assert rec_row_of == {2: 1, 5: 2}
    assert recs.dtype == np.int32


def test_build_lookup_matches_the_toy():
    recommendations, lookup = toy()
    _, rec_row_of = build_recs(recommendations, k=2)
    ids, rec_rows, buckets = build_lookup(lookup, rec_row_of)
    assert ids.tolist() == [3, 16, 255]                                 # sorted, for np.searchsorted
    assert rec_rows.tolist() == [0, 1, 2]                               # customer 7 has no history -> 0
    assert buckets.tolist() == [2, 0, 3]
    assert ids.dtype == np.uint64


def test_duplicate_64bit_id_is_refused():
    recommendations, lookup = toy()
    lookup.loc[2, "customer_id"] = "a" * 48 + "00000000000000ff"       # different id, SAME last 16 characters
    _, rec_row_of = build_recs(recommendations, k=2)
    with pytest.raises(ValueError, match="not unique"):
        build_lookup(lookup, rec_row_of)


def test_list_with_a_repeated_article_is_refused():
    recommendations = toy()[0]
    recommendations.loc[1, "article_id"] = 900                          # customer 5 gets [900, 900]
    with pytest.raises(ValueError, match="distinct"):
        build_recs(recommendations, k=2)


def test_bucket_lists_row_per_bucket():
    table = pd.DataFrame({"age_bucket": [1, 0, 1, 0], "rank": [1, 2, 2, 1], "article_id": [11, 2, 12, 1]})
    assert build_bucket_lists(table, k=2).tolist() == [[1, 2], [11, 12]]   # row 0 = global
