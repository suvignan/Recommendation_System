"""The Day 6 paper example (notebook 05 Step 2): bought 3 articles, my top 10 has 2 of them, at positions 1 and 7."""
from recsys.utils.metrics import ndcg_at_k, recall_at_k, score

RECOMMENDED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]        # made-up article numbers
BOUGHT = {1, 7, 99}                                   # 99 was bought but never recommended


def test_paper_example_recall():
    assert round(recall_at_k(RECOMMENDED, BOUGHT, 10), 4) == 0.6667          # 2 hits / 3 bought


def test_paper_example_ndcg():
    # DCG = 1/log2(2) + 1/log2(8) = 1.3333; IDCG = 1 + 0.6309 + 0.5 = 2.1309 -> 0.6257
    assert round(ndcg_at_k(RECOMMENDED, BOUGHT, 10), 4) == 0.6257


def test_perfect_list_ndcg_is_one():
    assert ndcg_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 10) == 1.0


def test_no_hits_is_zero():
    assert recall_at_k(RECOMMENDED, {50, 60}, 10) == 0.0
    assert ndcg_at_k(RECOMMENDED, {50, 60}, 10) == 0.0


def test_moving_a_hit_up_changes_ndcg_not_recall():
    moved = [1, 7, 2, 3, 4, 5, 6, 8, 9, 10]            # the hit at position 7 moved to position 2
    assert round(recall_at_k(moved, BOUGHT, 10), 4) == 0.6667                 # recall: still 2 of 3
    assert round(ndcg_at_k(moved, BOUGHT, 10), 4) == 0.7654                   # NDCG: (1 + 0.6309) / 2.1309


def test_score_returns_one_value_per_customer():
    recall, ndcg = score([RECOMMENDED, RECOMMENDED], [[1, 7, 99], [50, 60]])  # bought as lists, like the answer key
    assert recall.round(4).tolist() == [0.6667, 0.0]
    assert ndcg.round(4).tolist() == [0.6257, 0.0]
