"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces:

1. Boolean retrieval: given a query, treat it as an AND (conjunctive) or
   OR (disjunctive) combination of terms and return the matching document
   set — no ranking, just set membership.

2. Vector-space ranking: represent the query and each candidate document
   as TF-IDF weighted vectors and rank by cosine similarity:

       w(t, d) = tf(t, d) * log( N / df(t) )
       sim(q, d) = (q . d) / (||q|| * ||d||)

Both read from the same InvertedIndex built in indexer.py, using its
CSR-style postings arrays so the per-posting work happens in numpy rather
than in Python loops.
"""
import math
import numpy as np
from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize
from submission.ranking_utils import top_k

_INDEX: InvertedIndex = None
_IDF: np.ndarray = None        # float64, indexed by term ordinal: log(N / df)
_DOC_NORMS: np.ndarray = None  # float64, per doc: sqrt(sum_t (tf*idf)^2)


def build(index: InvertedIndex) -> None:
    """Precompute VSM-specific state (IDF per term, document vector norms)
    from the InvertedIndex built in indexer.py."""
    global _INDEX, _IDF, _DOC_NORMS
    _INDEX = index

    df = (index.post_off[1:] - index.post_off[:-1]).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        idf = np.log(np.where(df > 0, index.N / np.maximum(df, 1.0), 1.0))
    idf[df == 0] = 0.0
    _IDF = idf

    if index.post_doc.size == 0:
        _DOC_NORMS = np.zeros(index.N, dtype=np.float64)
        return

    # Expand each posting's term ordinal, then accumulate (tf*idf)^2 per doc.
    counts = (index.post_off[1:] - index.post_off[:-1]).astype(np.int64)
    post_term = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    w = index.post_tf.astype(np.float64) * _IDF[post_term]
    sq = np.bincount(index.post_doc, weights=w * w, minlength=index.N)
    _DOC_NORMS = np.sqrt(sq)


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its
    terms."""
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build(index) must be called before searching.")
    if mode not in {"and", "or"}:
        raise ValueError("mode must be either 'and' or 'or'")
    tokens = tokenize(query)
    if not tokens:
        return []

    doc_sets = [_INDEX.postings_for(term)[0] for term in tokens]

    result = doc_sets[0]
    for other in doc_sets[1:]:
        if mode == "and":
            result = np.intersect1d(result, other, assume_unique=True)
        else:
            result = np.union1d(result, other)

    return [_INDEX.doc_id_map[int(d)] for d in np.sort(result)]


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    TF-IDF cosine similarity, highest score first."""
    if _INDEX is None or k <= 0:
        return []
    tokens = tokenize(query)
    if not tokens:
        return []

    # Query vector: tf * idf per distinct term.
    q_tf = {}
    for t in tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    q_norm_sq = 0.0
    q_terms = []
    for term, tf in q_tf.items():
        tid = _INDEX.term_ids.get(term)
        if tid is None:
            continue
        w = tf * float(_IDF[tid])
        if w == 0.0:
            continue
        q_terms.append((tid, w))
        q_norm_sq += w * w

    if q_norm_sq == 0.0:
        return []
    q_norm = math.sqrt(q_norm_sq)

    acc = np.zeros(_INDEX.N, dtype=np.float64)
    touched = []
    for tid, q_w in q_terms:
        lo = int(_INDEX.post_off[tid])
        hi = int(_INDEX.post_off[tid + 1])
        if hi <= lo:
            continue
        docs = _INDEX.post_doc[lo:hi]
        d_w = _INDEX.post_tf[lo:hi].astype(np.float64) * float(_IDF[tid])
        acc[docs] += q_w * d_w
        touched.append(docs)

    if not touched:
        return []

    candidates = np.unique(np.concatenate(touched)) if len(touched) > 1 else touched[0]
    norms = _DOC_NORMS[candidates]
    keep = norms > 0.0
    candidates = candidates[keep]
    if candidates.size == 0:
        return []
    sims = acc[candidates] / (q_norm * norms[keep])

    # Deterministic tie-break: similarity descending, then doc id ascending.
    top_docs, top_sims = top_k(candidates, sims, k)

    return [
        (_INDEX.doc_id_map[int(doc)], float(s))
        for doc, s in zip(top_docs, top_sims)
    ]
