"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces to implement:

1. Boolean retrieval: given a query, treat it as an AND (conjunctive) or
   OR (disjunctive) combination of terms and return the matching document
   set — no ranking, just set membership. Useful as a fast candidate
   filter and as a sanity check ("does my index even find the right
   documents for this query?").

2. Vector-space ranking: represent the query and each candidate document
   as TF-IDF weighted vectors and rank by cosine similarity. A standard
   TF-IDF weight for term t in document d:

       w(t, d) = tf(t, d) * log( N / df(t) )

   (log base is your choice — just be consistent), and cosine similarity
   between query vector q and document vector d:

       sim(q, d) = (q . d) / (||q|| * ||d||)

Both pieces should read from the same InvertedIndex you build in
indexer.py.
"""
import math
import heapq
from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: InvertedIndex = None
_IDF_CACHE = {}
_DOC_NORMS = {}

def build(index: InvertedIndex) -> None:
    """Optional: precompute anything VSM-specific (e.g. document vector
    norms) from the InvertedIndex built in indexer.py.
    """
    global _INDEX, _IDF_CACHE, _DOC_NORMS
    _INDEX = index
    _IDF_CACHE = {}
    _DOC_NORMS = {}
    
    # IDF and tf-idf norms
    for term, encoded in index.postings.items():
        df = len(encoded) // 2
        idf = math.log(index.N / df)
        _IDF_CACHE[term] = idf
        
        doc_id = 0
        for i in range(0, len(encoded), 2):
            doc_id += encoded[i]
            tf = encoded[i+1]
            w = tf * idf
            if doc_id not in _DOC_NORMS:
                _DOC_NORMS[doc_id] = 0.0
            _DOC_NORMS[doc_id] += w * w
            
    for d in _DOC_NORMS:
        _DOC_NORMS[d] = math.sqrt(_DOC_NORMS[d])


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its
    terms."""
    global _INDEX
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build(index) must be called before searching.")
    if mode not in {"and", "or"}:
        raise ValueError("mode must be either 'and' or 'or'")
    tokens = tokenize(query)
    if not tokens:
        return []
        
    doc_sets = []
    for term in tokens:
        if term not in _INDEX.postings:
            doc_sets.append(set())
            continue
        encoded = _INDEX.postings[term]
        d_set = set()
        doc_id = 0
        for i in range(0, len(encoded), 2):
            doc_id += encoded[i]
            d_set.add(doc_id)
        doc_sets.append(d_set)
        
    if mode == "and":
        res = doc_sets[0]
        for s in doc_sets[1:]:
            res = res.intersection(s)
    else:
        res = doc_sets[0]
        for s in doc_sets[1:]:
            res = res.union(s)
            
    return [_INDEX.doc_id_map[d] for d in sorted(res)]


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    TF-IDF cosine similarity, highest score first."""
    global _INDEX, _IDF_CACHE, _DOC_NORMS
    if _INDEX is None or k <= 0:
        return []
    tokens = tokenize(query)
    if not tokens:
        return []
        
    # query vector tf
    q_tf = {}
    for t in tokens:
        q_tf[t] = q_tf.get(t, 0) + 1
        
    q_norm = 0.0
    q_weights = {}
    for t, tf in q_tf.items():
        if t in _IDF_CACHE:
            w = tf * _IDF_CACHE[t]
            q_weights[t] = w
            q_norm += w * w
            
    if q_norm == 0.0:
        return []
    q_norm = math.sqrt(q_norm)
    
    doc_scores = {}
    for t, q_w in q_weights.items():
        encoded = _INDEX.postings[t]
        doc_id = 0
        for i in range(0, len(encoded), 2):
            doc_id += encoded[i]
            d_tf = encoded[i+1]
            d_w = d_tf * _IDF_CACHE[t]
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += q_w * d_w
            
    res = []
    for doc_id, dot_prod in doc_scores.items():
        doc_norm = _DOC_NORMS[doc_id]
        if doc_norm == 0.0:
            continue
        sim = dot_prod / (q_norm * doc_norm)
        res.append((doc_id, sim))
        
    top_docs = heapq.nsmallest(k, res, key=lambda x: (-x[1], x[0]))
    return [(_INDEX.doc_id_map[d], float(s)) for d, s in top_docs]
