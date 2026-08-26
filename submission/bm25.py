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
"""
import math
import heapq
from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: InvertedIndex = None
_IDF_CACHE = {}


def build(index: InvertedIndex) -> None:
    """Optional: precompute anything BM25-specific (e.g. cached IDF values
    per term) from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    global _INDEX, _IDF_CACHE
    _INDEX = index
    _IDF_CACHE = {}
    
    # Precompute IDF for all terms in the index
    for term, encoded_postings in index.postings.items():
        df = len(encoded_postings) // 2
        idf = math.log((index.N - df + 0.5) / (df + 0.5) + 1.0)
        _IDF_CACHE[term] = idf


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""
    global _INDEX, _IDF_CACHE
    if _INDEX is None:
        raise RuntimeError("bm25.build(index) must be called before score().")
    if k <= 0 or _INDEX.N == 0 or _INDEX.avg_doc_len == 0.0:
        return []
    tokens = tokenize(query)
    
    doc_scores = {}
    
    for term in tokens:
        if term not in _INDEX.postings:
            continue
            
        idf = _IDF_CACHE.get(term, 0.0)
        
        # Decode postings
        encoded = _INDEX.postings[term]
        doc_id = 0
        for i in range(0, len(encoded), 2):
            doc_id += encoded[i]
            tf = encoded[i+1]
            
            # Compute BM25 term weight
            doc_len = _INDEX.doc_len[doc_id]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / _INDEX.avg_doc_len))
            
            weight = idf * (numerator / denominator)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += weight
            
    # O(C log k), not O(C log C), for C matching candidate documents.
    top_docs = heapq.nsmallest(k, doc_scores.items(), key=lambda x: (-x[1], x[0]))
    return [(_INDEX.doc_id_map[doc_id], float(sc)) for doc_id, sc in top_docs]
