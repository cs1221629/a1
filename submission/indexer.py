"""
submission/indexer.py — build your inverted index here.

This is one of the required components (assignment Section 4.1): you must
build the inverted index yourself, without an existing search/indexing
library (Lucene, Elasticsearch, Pyserini, Whoosh, etc.).

A `tokenize()` helper is provided below purely so that tokenization is
consistent across your Boolean/VSM and BM25 scorers —
feel free to replace it (e.g. add stemming or stopword removal), just make
sure every scorer that reads this index was built with the same tokenizer.

Everything else — the postings representation, what per-document and
collection statistics you track, whether you add positions for
proximity/phrase features — is your design decision. `InvertedIndex`
below sketches a minimal, obviously-sufficient shape; you do not have to
use it, but if you do, filling in `build()` and `document_frequency()` is
enough to support Boolean/VSM and BM25.

Persistence (assignment Section 4.1 / Section 7 "index size" scoring):
`build_index()` in retrieve.py runs in one process and `load_index()` runs
in a separate, later one — so whatever this index needs at query time must
round-trip through `save()`/`load()` below, not just live as Python
attributes. The on-disk byte size of what `save()` writes is graded
directly (smaller, relative to the class median, scores better), so a
compact postings encoding is worth more here than in most course
assignments — see the `save()` docstring for concrete starting points.
"""
import re
from functools import lru_cache
from typing import Dict, List, Tuple
import os
import pickle
import zlib
from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "has", "have", "how",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "there", "these", "this", "to", "was", "were", "what", "when", "which", "who",
    "will", "with", "would", "any", "also", "among", "about", "than", "then",
})
_STEMMER = PorterStemmer()


@lru_cache(maxsize=300_000)
def _stem(token: str) -> str:
    """Cache stemming because corpus vocabularies repeat tokens heavily."""
    return _STEMMER.stem(token)


def tokenize(text: str) -> List[str]:
    """Lowercase, remove ordinary English stopwords, and Porter-stem.

    This deliberately has no corpus- or topic-specific vocabulary rules, so
    the indexer generalizes to any English collection in the assignment.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [_stem(token) for token in tokens if token not in _STOPWORDS and len(token) > 1]


class InvertedIndex:
    """A minimal inverted index skeleton. Extend the data structures here
    however your design needs (e.g. term positions for phrase/proximity
    scoring, a more compact postings representation for the efficiency
    bonus) — this is a starting point, not a fixed schema.
    """

    def __init__(self):
        # We will use integer doc IDs internally for efficiency.
        self.doc_id_map: List[str] = [] # int -> str
        self.doc_len: List[int] = []    # int -> length
        
        # self.postings maps a term to a flat list of integers:
        # [doc_id0, tf0, delta_doc_id1, tf1, delta_doc_id2, tf2, ...]
        self.postings: Dict[str, List[int]] = {}
        
        self.N: int = 0
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus().
        """
        # Permit reuse of an index object without leaking a previous build.
        self.doc_id_map = []
        self.doc_len = []
        self.postings = {}
        self.N = len(corpus)
        total_len = 0
        
        # Temporary structure for building postings: term -> {doc_id_int: tf}
        temp_postings: Dict[str, Dict[int, int]] = {}
        
        for doc_idx, (doc_id, text) in enumerate(corpus):
            self.doc_id_map.append(doc_id)
            tokens = tokenize(text)
            doc_length = len(tokens)
            self.doc_len.append(doc_length)
            total_len += doc_length
            
            for token in tokens:
                if token not in temp_postings:
                    temp_postings[token] = {}
                if doc_idx not in temp_postings[token]:
                    temp_postings[token][doc_idx] = 0
                temp_postings[token][doc_idx] += 1
                
        self.avg_doc_len = total_len / self.N if self.N > 0 else 0.0
        
        # Convert to delta-encoded lists
        for term, term_docs in temp_postings.items():
            encoded = []
            last_doc_id = 0
            # sort by doc_id to enable delta encoding
            for doc_id_int in sorted(term_docs.keys()):
                tf = term_docs[doc_id_int]
                encoded.append(doc_id_int - last_doc_id)
                encoded.append(tf)
                last_doc_id = doc_id_int
            self.postings[term] = encoded

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        if term not in self.postings:
            return 0
        # The list has 2 entries per document (delta_doc_id, tf)
        return len(self.postings[term]) // 2

    def save(self, index_dir: str) -> None:
        """Persist everything document_frequency() / your scorers need to
        `index_dir`, so `load()` can reconstruct this object in a fresh
        process with no memory of `build()` ever having run. Called from
        retrieve.build_index().
        """
        os.makedirs(index_dir, exist_ok=True)
        path = os.path.join(index_dir, "index.bin")
        data = {
            "doc_id_map": self.doc_id_map,
            "doc_len": self.doc_len,
            "postings": self.postings,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len
        }
        compressed = zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        with open(path, "wb") as f:
            f.write(compressed)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct an InvertedIndex purely from what save() wrote to
        `index_dir`. Called in a fresh process — do not rely on any state
        other than what's actually on disk in `index_dir`.
        """
        path = os.path.join(index_dir, "index.bin")
        with open(path, "rb") as f:
            compressed = f.read()
            
        data = pickle.loads(zlib.decompress(compressed))
        index = cls()
        index.doc_id_map = data["doc_id_map"]
        index.doc_len = data["doc_len"]
        index.postings = data["postings"]
        index.N = data["N"]
        index.avg_doc_len = data["avg_doc_len"]
        return index

