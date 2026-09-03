"""
submission/custom_scorer.py — optional combined/custom scorer.

Not required, but explicitly called out in the assignment (Section 4.1) as
"any linear or non-linear combination of your Boolean/VSM and BM25
signals ... this is where separation in the leaderboard tends to happen".

What this implements: a linear blend of the two ranking signals the
assignment already requires, over the union of the documents either of
them retrieves,

    blended(d) = lam * norm(bm25(d)) + (1 - lam) * norm(cosine(d))

with `lam = 1.0` degenerating exactly to plain BM25. BM25 scores are
unbounded sums of IDF-weighted term contributions while cosine
similarities live in [0, 1], so the two are min-max normalised *per query*
across that query's candidate set before blending — otherwise BM25's scale
would swamp the cosine term and `lam` would be meaningless.

Both signals are accumulated in a single pass over each query term's
postings (one gather per term feeding two accumulators), so blending costs
roughly one extra array add per posting rather than running two
independent scorers back to back.

Whether this actually ships is decided empirically, not by preference —
see `scripts/tune.py --stage blend` and the shipping rule in
submission/retrieve.py.
"""
import math
import numpy as np
from typing import List, Tuple

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex, tokenize
from submission.ranking_utils import CandidateSet, top_k

_INDEX: InvertedIndex = None
_ACC_BM: np.ndarray = None
_ACC_VSM: np.ndarray = None
_VSM_W: np.ndarray = None        # float64, per posting: tf * idf
_CANDS = None                    # ranking_utils.CandidateSet, length N


def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(), not retrieve.build_index() — the
    harness runs those two in separate processes. Anything this needs at
    query time either comes from the loaded InvertedIndex or must have
    been written to index_dir by InvertedIndex.save() (which then counts
    toward your index-size score).

    Depends on bm25.build() and boolean_vsm.build() having run for their
    cached IDF tables and document norms; both are idempotent, so call
    them here rather than relying on ordering elsewhere.
    """
    global _INDEX, _ACC_BM, _ACC_VSM, _VSM_W, _CANDS
    _INDEX = index
    bm25.build(index)
    boolean_vsm.build(index)
    _ACC_BM = np.zeros(index.N, dtype=np.float32)
    _ACC_VSM = np.zeros(index.N, dtype=np.float64)
    _CANDS = CandidateSet(index.N)

    # Per-posting tf*idf, precomputed once instead of rebuilt per query
    # term. score() previously did `post_tf[lo:hi].astype(np.float64) *
    # idf`, which allocates two temporaries for every term of every query;
    # hoisting it here is the same arithmetic in the same order, so scores
    # are bit-identical, but the query loop becomes a single multiply.
    counts = (index.post_off[1:] - index.post_off[:-1]).astype(np.int64)
    post_term = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    _VSM_W = index.post_tf.astype(np.float64) * boolean_vsm._IDF[post_term]


def _normalize(values: np.ndarray) -> np.ndarray:
    """Min-max a score vector onto [0, 1]; a flat vector maps to all zeros."""
    lo = values.min()
    hi = values.max()
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def score(query: str, k: int, lam: float = 0.9, k1: float = 2.0,
          b: float = 0.6, delta: float = 0.0) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by the
    blended BM25 / cosine-VSM score, highest score first."""
    if _INDEX is None:
        raise RuntimeError("custom_scorer.build(index) must be called before score().")
    if k <= 0 or _INDEX.N == 0 or _INDEX.avg_doc_len == 0.0:
        return []
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    bm25.configure(k1, b)

    tokens = tokenize(query)
    if not tokens:
        return []

    q_tf = {}
    for t in tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    touched = False
    q_norm_sq = 0.0
    for term, tf_q in q_tf.items():
        tid = _INDEX.term_ids.get(term)
        if tid is None:
            continue
        lo = int(_INDEX.post_off[tid])
        hi = int(_INDEX.post_off[tid + 1])
        if hi <= lo:
            continue
        docs = _INDEX.post_doc[lo:hi]

        # BM25 side, using the impact weights bm25.configure() precomputed.
        # Multiplied by the query-side term frequency so that a term
        # repeated in the query counts once per occurrence, exactly as
        # bm25.score() does when it loops over the token list. Without
        # this, lam=1.0 would not reduce to plain BM25 and the blend
        # sweep would be comparing against the wrong reference point.
        idf_bm = tf_q * float(bm25._IDF[tid])
        if delta:
            _ACC_BM[docs] += idf_bm * (delta + bm25._WEIGHT[lo:hi])
        else:
            _ACC_BM[docs] += idf_bm * bm25._WEIGHT[lo:hi]

        # VSM side: tf-idf dot product against the query vector.
        idf_vsm = float(boolean_vsm._IDF[tid])
        q_w = tf_q * idf_vsm
        q_norm_sq += q_w * q_w
        _ACC_VSM[docs] += q_w * _VSM_W[lo:hi]

        _CANDS.add(docs)
        touched = True

    if not touched:
        return []

    candidates = _CANDS.collect()
    bm_scores = _ACC_BM[candidates].astype(np.float64)

    doc_norms = boolean_vsm._DOC_NORMS[candidates]
    q_norm = math.sqrt(q_norm_sq)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(doc_norms > 0.0,
                       _ACC_VSM[candidates] / (q_norm * np.maximum(doc_norms, 1e-12)),
                       0.0)

    blended = lam * _normalize(bm_scores) + (1.0 - lam) * _normalize(cos)

    top_docs, top_scores = top_k(candidates, blended, k)

    _ACC_BM[candidates] = 0.0
    _ACC_VSM[candidates] = 0.0
    _CANDS.clear(candidates)

    return [
        (_INDEX.doc_id_map[int(doc)], float(s))
        for doc, s in zip(top_docs, top_scores)
    ]
