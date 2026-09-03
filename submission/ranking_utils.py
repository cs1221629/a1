"""
submission/ranking_utils.py — shared top-k selection for the scorers.

Small, non-graded helper factored out of bm25.py / boolean_vsm.py /
custom_scorer.py so all three rank identically and break ties the same
way.

Why this exists rather than a bare `np.argpartition`: argpartition only
guarantees that the k-th element is in its final position, and it makes
*no* guarantee about which of several equally-scored elements end up on
the winning side of that split. When documents tie exactly at the k-th
score — which happens on this corpus, e.g. two documents of identical
length matching the same query terms with the same frequencies — a bare
argpartition silently keeps an arbitrary one of them, so the intended
"lowest doc id wins" tie-break never gets a chance to run. That produces
rankings that differ run-to-run against the same index.

`top_k` below fixes that by re-including every candidate tied with the
k-th score before applying the deterministic (-score, doc_id) ordering.
"""
import numpy as np
from typing import Tuple


class CandidateSet:
    """Collects the union of the posting lists a query touched.

    The obvious way to do this is `np.unique(np.concatenate(touched))`, but
    that sorts every posting the query visited — on this corpus a handful
    of common query terms means sorting one to two million doc ids, which
    measured at 4.5 ms of a 12.5 ms query, more than a third of the whole
    latency budget.

    Marking a byte per touched doc and then scanning the mark array instead
    is O(N) in the collection size (171k) rather than O(P log P) in the
    postings visited, and it returns doc ids already ascending and
    deduplicated — exactly the contract `top_k` wants. Reset costs one
    scatter over the candidates rather than a fresh allocation, so the
    array is allocated once per index and reused for every query.
    """

    __slots__ = ("_mask",)

    def __init__(self, num_docs: int):
        self._mask = np.zeros(num_docs, dtype=np.uint8)

    def add(self, docs: np.ndarray) -> None:
        self._mask[docs] = 1

    def collect(self) -> np.ndarray:
        return np.flatnonzero(self._mask)

    def clear(self, candidates: np.ndarray) -> None:
        self._mask[candidates] = 0


def top_k(candidates: np.ndarray, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the top `k` (candidates, scores), ordered by score descending
    and then by candidate id ascending.

    `candidates` must be sorted ascending (doc ids). `scores[i]` is the
    score of `candidates[i]`.
    """
    if candidates.size == 0 or k <= 0:
        empty_i = np.zeros(0, dtype=candidates.dtype)
        return empty_i, np.zeros(0, dtype=scores.dtype)

    k_eff = min(k, candidates.size)

    if k_eff < candidates.size:
        sel = np.argpartition(-scores, k_eff - 1)[:k_eff]
        threshold = scores[sel].min()
        # Everything strictly better than the k-th score is definitely in.
        strictly_better = np.flatnonzero(scores > threshold)
        remaining = k_eff - strictly_better.size
        if remaining > 0:
            # Fill the rest from the docs tied at the threshold. `candidates`
            # is doc-id ascending, so flatnonzero returns them in doc-id
            # order and the lowest doc ids win the tie deterministically.
            tied = np.flatnonzero(scores == threshold)[:remaining]
            sel = np.concatenate((strictly_better, tied))
        else:
            sel = strictly_better[:k_eff]
    else:
        sel = np.arange(candidates.size)

    top_ids = candidates[sel]
    top_scores = scores[sel]
    # Deterministic final ordering: score descending, then doc id ascending.
    order = np.lexsort((top_ids, -top_scores))
    return top_ids[order], top_scores[order]
