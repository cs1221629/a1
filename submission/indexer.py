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
from typing import Dict, Iterable, List, Tuple
import os
import pickle
import zlib
import array
import struct
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
        self.doc_len: array.array = array.array('I')  # unsigned int

        # self.postings maps a term to an array of integers:
        # [doc_id0, tf0, delta_doc_id1, tf1, delta_doc_id2, tf2, ...]
        self.postings: Dict[str, array.array] = {}

        self.N: int = 0
        self.avg_doc_len: float = 0.0
        # Pre-computed doc_len[i] / avg_doc_len for BM25 speedup
        self.doc_len_ratio: array.array = array.array('f')

    def build(self, corpus: Iterable[Tuple[str, str]]) -> None:
        """corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus().
        """
        # Permit reuse of an index object without leaking a previous build.
        self.doc_id_map = []
        doc_lens = []
        self.postings = {}
        self.N = 0
        total_len = 0

        # Temporary structure for building postings: term -> {doc_id_int: tf}
        temp_postings: Dict[str, Dict[int, int]] = {}

        for doc_idx, (doc_id, text) in enumerate(corpus):
            self.doc_id_map.append(doc_id)
            self.N += 1
            tokens = tokenize(text)
            doc_length = len(tokens)
            doc_lens.append(doc_length)
            total_len += doc_length

            for token in tokens:
                if token not in temp_postings:
                    temp_postings[token] = {}
                if doc_idx not in temp_postings[token]:
                    temp_postings[token][doc_idx] = 0
                temp_postings[token][doc_idx] += 1

        self.avg_doc_len = total_len / self.N if self.N > 0 else 0.0
        self.doc_len = array.array('I', doc_lens)

        # Pre-compute doc_len / avg_doc_len for BM25 scoring speed
        if self.avg_doc_len > 0:
            self.doc_len_ratio = array.array('f',
                (dl / self.avg_doc_len for dl in doc_lens))
        else:
            self.doc_len_ratio = array.array('f', [0.0] * self.N)

        # Convert to delta-encoded arrays
        for term, term_docs in temp_postings.items():
            encoded = array.array('I')
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
        # Convert array.array postings to bytes for more compact pickle
        postings_bytes = {term: _uint_array_to_bytes(arr)
                          for term, arr in self.postings.items()}
        data = {
            "doc_id_map": self.doc_id_map,
            "doc_len": _uint_array_to_bytes(self.doc_len),
            "postings": postings_bytes,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len
        }
        compressed = zlib.compress(
            pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), level=6)
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
        index.doc_len = _bytes_to_uint_array(data["doc_len"])
        # Reconstruct postings arrays from bytes
        index.postings = {}
        for term, raw in data["postings"].items():
            index.postings[term] = _bytes_to_uint_array(raw)
        index.N = data["N"]
        index.avg_doc_len = data["avg_doc_len"]
        # Recompute doc_len_ratio (cheap, avoids persisting it)
        if index.avg_doc_len > 0:
            index.doc_len_ratio = array.array('f',
                (dl / index.avg_doc_len for dl in index.doc_len))
        else:
            index.doc_len_ratio = array.array('f', [0.0] * index.N)
        return index


def _uint_array_to_bytes(values: array.array) -> bytes:
    """Encode unsigned integers with a fixed four-byte little-endian format."""
    encoded = bytearray(len(values) * 4)
    for offset, value in enumerate(values):
        struct.pack_into("<I", encoded, offset * 4, value)
    return bytes(encoded)


def _bytes_to_uint_array(raw: bytes) -> array.array:
    """Decode the portable uint32 format into the index's native array type."""
    if len(raw) % 4:
        raise ValueError("corrupt index: uint32 data has an invalid length")
    return array.array("I", (value[0] for value in struct.iter_unpack("<I", raw)))

