"""Recall@k and NDCG@k: the ONE copy of the Day 6 metrics (the notebooks had copies; the pipeline uses this file)."""
import math

import numpy as np


def recall_at_k(recommended, bought, k):
    """Hits in the first k / number of articles bought. Recall ignores WHERE the hits are."""
    if len(bought) == 0:
        return 0.0
    hits = len(set(recommended[:k]) & set(bought))
    return hits / len(bought)


def ndcg_at_k(recommended, bought, k):
    """DCG / ideal DCG: a hit at position p is worth 1 / log2(p + 1), so hits near the top count more."""
    ideal_hits = min(len(bought), k)
    if ideal_hits == 0:
        return 0.0

    dcg = 0.0
    for index, article in enumerate(recommended[:k]):       # index 0 = position 1 -> log2(index + 2)
        if article in bought:
            dcg += 1 / math.log2(index + 2)

    idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))   # all hits at the top
    return dcg / idcg


def score(recommendations, bought_lists, k=10):
    """One recall and one NDCG PER CUSTOMER (arrays), so the caller can average all / warm / cold separately."""
    recall_scores = []
    ndcg_scores = []
    for recommended, bought in zip(recommendations, bought_lists):
        recommended = list(recommended)         # numpy row or list -> list
        bought = set(bought)                    # set: 'in' is instant (an array would be searched element by element)
        recall_scores.append(recall_at_k(recommended, bought, k))
        ndcg_scores.append(ndcg_at_k(recommended, bought, k))
    return np.array(recall_scores), np.array(ndcg_scores)
