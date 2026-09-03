"""
submission/lm_dirichlet.py — query-likelihood retrieval with Dirichlet
smoothing.

Why a third scorer at all: `custom_scorer.py` fuses signals, and fusion
only pays when the signals disagree. BM25 and cosine TF-IDF are both
tf-idf-shaped and agree most of the time, which caps what blending them
can buy. A query-likelihood language model is derived from a genuinely
different place — it asks "how likely is this query under a model of this
document?" rather than "how well do these weight vectors align?" — and its
length normalisation falls out of the smoothing rather than being a tuned
`b`. That makes it the standard third leg of a sparse fusion run.

The model (Zhai & Lafferty, "A Study of Smoothing Methods for Language
Models Applied to Ad Hoc Information Retrieval"):

    score(d, q) = sum_t  c(t, q) * log( 1 + tf(t, d) / (mu * p(t|C)) )
                  + |q| * log( mu / (|d| + mu) )

with the collection model p(t|C) = cf(t) / sum_t cf(t), where cf(t) is the
number of *occurrences* of t across the collection (not its document
frequency). mu is the smoothing mass, conventionally 1000-2500.

The second term does not depend on the query term, only on the document,
so it is precomputed per document and applied once to the candidate set
rather than inside the per-term loop.

Nothing here is persisted: cf(t) is `add.reduceat` over the postings that
`InvertedIndex.load()` already reconstructs, so this scorer costs zero
bytes of index size and its setup lands in load time, which the leaderboard
measures but does not score.
"""
import numpy as np
from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize
from submission.ranking_utils import CandidateSet, top_k

_INDEX: InvertedIndex = None
_LOG_PC: np.ndarray = None       # float64 per term: log(mu * p(t|C)) at _MU
_WEIGHT: np.ndarray = None       # float32 per posting: log(1 + tf/(mu*p(t|C)))
_LEN_NORM: np.ndarray = None     # float64 per doc: log(mu / (|d| + mu))
_CF: np.ndarray = None           # float64 per term: collection frequency
_MU = None
_ACC: np.ndarray = None
_CANDS = None


def build(index: InvertedIndex) -> None:
    """Precompute the collection model from the loaded index.

    Call from retrieve.load_index(), like the other scorers' build().
    """
    global _INDEX, _CF, _WEIGHT, _LEN_NORM, _MU, _ACC, _CANDS
    _INDEX = index
    _WEIGHT = None
    _LEN_NORM = None
    _MU = None
    _ACC = np.zeros(index.N, dtype=np.float32)
    _CANDS = CandidateSet(index.N)

    num_terms = index.post_off.size - 1
    if index.post_doc.size == 0 or num_terms == 0:
        _CF = np.zeros(max(num_terms, 0), dtype=np.float64)
        return

    # Collection frequency per term: sum of tf over each term's postings
    # block. reduceat needs the block starts, and it must not be handed a
    # start equal to the array length, so empty trailing terms are summed
    # separately and zeroed.
    starts = index.post_off[:-1]
    counts = index.post_off[1:] - starts
    safe = np.minimum(starts, index.post_tf.size - 1)
    cf = np.add.reduceat(index.post_tf.astype(np.float64), safe)
    cf[counts == 0] = 0.0
    _CF = cf


def configure(mu: float) -> None:
    """Precompute the per-posting and per-document terms for one mu."""
    global _WEIGHT, _LEN_NORM, _MU
    if _INDEX is None:
        raise RuntimeError("lm_dirichlet.build(index) must be called before configure().")
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if _MU == mu:
        return

    total = _CF.sum()
    if total <= 0.0 or _INDEX.post_doc.size == 0:
        _WEIGHT = np.zeros(_INDEX.post_doc.size, dtype=np.float32)
        _LEN_NORM = np.zeros(_INDEX.N, dtype=np.float64)
        _MU = mu
        return

    # mu * p(t|C), expanded to one entry per posting.
    counts = (_INDEX.post_off[1:] - _INDEX.post_off[:-1]).astype(np.int64)
    post_term = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    mu_pc = mu * (_CF / total)
    # A term present in the index always has cf > 0, so the division is
    # safe; the maximum guards a degenerate index rather than real data.
    _WEIGHT = np.log1p(
        _INDEX.post_tf.astype(np.float64) / np.maximum(mu_pc[post_term], 1e-12)
    ).astype(np.float32)

    _LEN_NORM = np.log(mu / (_INDEX.doc_len.astype(np.float64) + mu))
    _MU = mu


def raw_scores(query: str, mu: float = 1000.0):
    """Return (candidates, scores) for `query` without truncating to k.

    Split out from score() so custom_scorer can fuse this signal with the
    others over a shared candidate set instead of paying for two top-k
    selections.
    """
    if _INDEX is None:
        raise RuntimeError("lm_dirichlet.build(index) must be called before scoring.")
    empty = np.zeros(0, dtype=np.int64)
    if _INDEX.N == 0 or _INDEX.post_doc.size == 0:
        return empty, np.zeros(0, dtype=np.float64)
    configure(mu)

    tokens = tokenize(query)
    if not tokens:
        return empty, np.zeros(0, dtype=np.float64)

    q_tf = {}
    for token in tokens:
        q_tf[token] = q_tf.get(token, 0) + 1

    touched = False
    matched_qlen = 0
    for term, tf_q in q_tf.items():
        tid = _INDEX.term_ids.get(term)
        if tid is None:
            continue
        lo = int(_INDEX.post_off[tid])
        hi = int(_INDEX.post_off[tid + 1])
        if hi <= lo:
            continue
        docs = _INDEX.post_doc[lo:hi]
        _ACC[docs] += tf_q * _WEIGHT[lo:hi]
        _CANDS.add(docs)
        matched_qlen += tf_q
        touched = True

    if not touched:
        return empty, np.zeros(0, dtype=np.float64)

    candidates = _CANDS.collect()
    # |q| here counts only the query terms the collection actually knows
    # about. Out-of-vocabulary terms contribute the same constant to every
    # document, so including them would shift all scores equally and change
    # nothing about the ranking, but it would make scores from different
    # queries harder to compare.
    scores = _ACC[candidates].astype(np.float64) + matched_qlen * _LEN_NORM[candidates]

    _ACC[candidates] = 0.0
    _CANDS.clear(candidates)
    return candidates, scores


def score(query: str, k: int, mu: float = 1000.0) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    Dirichlet-smoothed query likelihood, highest score first."""
    if k <= 0:
        return []
    candidates, scores = raw_scores(query, mu)
    if candidates.size == 0:
        return []
    top_docs, top_scores = top_k(candidates, scores, k)
    return [
        (_INDEX.doc_id_map[int(doc)], float(sc))
        for doc, sc in zip(top_docs, top_scores)
    ]
