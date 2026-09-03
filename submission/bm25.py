"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." See the assignment background (Section 3) for the
Robertson & Walker / Robertson & Zaragoza references this is based on.

BM25 score for a query Q = q1...qn against document D:

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )

A standard IDF variant (Robertson-Sparck Jones, +1-smoothed so it stays
non-negative even for terms occurring in more than half the corpus):

    IDF(qi) = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

where:
    N        = number of documents in the corpus
    df(qi)   = number of documents containing qi
    tf(qi,D) = term frequency of qi in D
    |D|      = length of D in tokens
    avgdl    = average document length across the corpus

k1 (typically 1.2-2.0) controls term-frequency saturation; b (in [0, 1])
controls document-length normalisation strength. Both must be exposed as
parameters, not hard-coded — you need to sweep them for your report
(assignment Section 8, "parameter search procedure for k1, b").

Scoring is vectorized with numpy against the CSR-style postings in
indexer.py: for each query term we take that term's postings slice and do
`acc[docs] += idf * weight` over the whole slice at once, instead of
walking postings and updating a Python dict per document. `configure(k1,
b)` precomputes the per-posting weight `tf*(k1+1) / (tf + K[doc])` once
per (k1, b) pair, so a query costs one gather + scaled add per term.
"""
import numpy as np
from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize
from submission.ranking_utils import CandidateSet, top_k

_INDEX: InvertedIndex = None
_IDF: np.ndarray = None          # float32, indexed by term ordinal
_WEIGHT: np.ndarray = None       # float32, per posting: tf*(k1+1)/(tf+K[doc])
_NORMALIZER_PARAMS = None
_ACC: np.ndarray = None          # persistent float32 accumulator, length N
_CANDS = None                    # ranking_utils.CandidateSet, length N


def build(index: InvertedIndex) -> None:
    """Precompute per-term IDF from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    global _INDEX, _IDF, _WEIGHT, _NORMALIZER_PARAMS, _ACC, _CANDS
    _INDEX = index
    _WEIGHT = None
    _NORMALIZER_PARAMS = None
    _CANDS = CandidateSet(index.N)

    df = (index.post_off[1:] - index.post_off[:-1]).astype(np.float64)
    _IDF = np.log((index.N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    _ACC = np.zeros(index.N, dtype=np.float32)


def configure(k1: float, b: float) -> None:
    """Precompute the per-posting BM25 weight for one (k1, b) parameter set."""
    global _WEIGHT, _NORMALIZER_PARAMS
    if _INDEX is None:
        raise RuntimeError("bm25.build(index) must be called before configure().")
    params = (k1, b)
    if _NORMALIZER_PARAMS == params:
        return
    # K depends only on the document and (k1, b), never on the query term.
    K = k1 * (1.0 - b + b * _INDEX.doc_len_ratio.astype(np.float64))
    tf = _INDEX.post_tf.astype(np.float64)
    _WEIGHT = (tf * (k1 + 1.0) / (tf + K[_INDEX.post_doc])).astype(np.float32)
    _NORMALIZER_PARAMS = params


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75,
          delta: float = 0.0) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""
    global _INDEX, _IDF, _WEIGHT, _ACC, _CANDS
    if _INDEX is None:
        raise RuntimeError("bm25.build(index) must be called before score().")
    if k <= 0 or _INDEX.N == 0 or _INDEX.avg_doc_len == 0.0:
        return []
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    configure(k1, b)
    tokens = tokenize(query)
    if not tokens:
        return []

    touched = False
    for term in tokens:
        tid = _INDEX.term_ids.get(term)
        if tid is None:
            continue
        lo = int(_INDEX.post_off[tid])
        hi = int(_INDEX.post_off[tid + 1])
        if hi <= lo:
            continue
        docs = _INDEX.post_doc[lo:hi]
        # BM25+; delta=0.0 is ordinary BM25. Ordinary BM25 is the shipped
        # configuration, and `delta + _WEIGHT[lo:hi]` would materialise a
        # second full-length temporary per query term purely to add zero,
        # so that case takes the branch without it.
        # Doc ids are unique within a term's posting block, so a plain
        # fancy-index += is correct here (no repeated-index collisions).
        idf = float(_IDF[tid])
        if delta:
            _ACC[docs] += idf * (delta + _WEIGHT[lo:hi])
        else:
            _ACC[docs] += idf * _WEIGHT[lo:hi]
        _CANDS.add(docs)
        touched = True

    if not touched:
        return []

    candidates = _CANDS.collect()
    scores = _ACC[candidates]

    # Deterministic tie-break (score descending, then doc id ascending),
    # including for documents tied exactly at the k-th score — see
    # ranking_utils.top_k for why a bare argpartition is not enough.
    top_docs, top_scores = top_k(candidates, scores, k)

    _ACC[candidates] = 0.0
    _CANDS.clear(candidates)

    return [
        (_INDEX.doc_id_map[int(doc)], float(sc))
        for doc, sc in zip(top_docs, top_scores)
    ]
