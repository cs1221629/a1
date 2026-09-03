# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""
submission/_tokenizer.pyx — the indexing hot loop, in C++.

OPTIONAL. `submission/indexer.py` imports this inside a try/except and
falls back to a pure-Python path that produces an identical index if the
extension is missing, so a machine with no compiler still works. The
assignment permits this explicitly (Section 4.1 and Section 10,
"Compiled extensions"): the indexing logic below is our own, the rule
being about not importing Lucene/Elasticsearch/Pyserini/Whoosh.

WHAT IT REPLACES AND WHY

The pure-Python streaming pass does three things per document:

    tokens   = text.encode(...).translate(TABLE).split()   # ~1.5 s total
    ordinals = list(map(raw_vocab.get, tokens))            # ~5.5 s total
    chunks.append(array.array("i", ordinals))              # ~1.7 s total

all of which are already C-speed calls — but they pay for materialising
29 million Python bytes objects, hashing each one, and boxing 29 million
ints. None of those objects outlive the loop; the only thing kept is an
integer per token. Doing the scan in C++ against an unordered_map keeps
the same work and skips the allocation entirely.

Tokenization here is *exactly* equivalent to the Python path: fold A-Z
onto a-z, keep [a-z0-9], treat every other byte as a separator. That is
the same rule as the 256-entry translate table followed by `split()`,
including for multi-byte UTF-8 (no continuation byte is in [a-z0-9], so
those become separators either way). Stopwords, the minimum-length rule
and stemming are deliberately NOT done here — they are applied once per
distinct vocabulary entry back in indexer.py, which is both faster and
keeps the tunable policy in Python where scripts/tune.py can reach it.

Equivalence is not asserted, it is tested: tests/test_fast_tokenizer.py
builds the same corpus both ways and compares every array.
"""
from libc.stdint cimport int32_t
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement

import numpy as np


cdef class VocabBuilder:
    """Accumulates a corpus-wide vocabulary and a flat stream of term
    ordinals, one entry per token occurrence, in document order.

    Usage mirrors the Python loop it replaces::

        vb = VocabBuilder()
        for doc_id, text in corpus:
            n_tokens = vb.add(text.encode("utf-8", "ignore"))
        vocab, ordinals = vb.finish()

    `vocab` is a list of bytes indexed by ordinal; `ordinals` is an int32
    numpy array of length sum(n_tokens).
    """

    cdef unordered_map[string, int] _ids
    cdef vector[string] _terms
    cdef vector[int32_t] _ordinals
    cdef string _tok

    def __cinit__(self):
        self._ordinals.reserve(1 << 22)

    cdef inline int _intern(self) noexcept:
        """Ordinal for the token currently in self._tok, inserting it on
        first sight. First-seen numbering matches the Python dict exactly;
        it does not actually matter, because indexer.py relabels term
        ordinals into lexicographic order at the end of build()."""
        cdef unordered_map[string, int].iterator it = self._ids.find(self._tok)
        cdef int ordinal
        if it == self._ids.end():
            ordinal = <int>self._terms.size()
            self._ids[self._tok] = ordinal
            self._terms.push_back(self._tok)
        else:
            ordinal = dereference(it).second
        return ordinal

    def add(self, bytes data):
        """Scan one document, appending an ordinal per token. Returns the
        number of tokens found, which is the document's raw length."""
        cdef const unsigned char* p = <const unsigned char*><const char*>data
        cdef Py_ssize_t n = len(data)
        cdef Py_ssize_t i
        cdef unsigned int c
        cdef Py_ssize_t count = 0

        self._tok.clear()
        for i in range(n):
            c = p[i]
            if 65 <= c <= 90:        # A-Z -> a-z
                c += 32
            if (97 <= c <= 122) or (48 <= c <= 57):
                self._tok.push_back(<char>c)
            elif not self._tok.empty():
                self._ordinals.push_back(<int32_t>self._intern())
                count += 1
                self._tok.clear()

        if not self._tok.empty():
            self._ordinals.push_back(<int32_t>self._intern())
            count += 1
            self._tok.clear()

        return count

    def finish(self):
        """Return (vocab_list, ordinals_int32_array) and release the
        builder's memory. The vocabulary is a list of bytes indexed by
        ordinal, matching the keys the Python path would have produced."""
        cdef Py_ssize_t m = self._ordinals.size()
        cdef Py_ssize_t k = self._terms.size()

        ordinals = np.empty(m, dtype=np.int32)
        cdef int32_t[::1] out = ordinals
        if m:
            memcpy(&out[0], self._ordinals.data(), <size_t>m * sizeof(int32_t))

        vocab = [None] * k
        cdef Py_ssize_t j
        cdef string term
        for j in range(k):
            # Assigning a std::string to an untyped Python target is the
            # documented Cython conversion to bytes; an explicit <bytes>
            # cast is a C cast and does not compile.
            term = self._terms[j]
            vocab[j] = term

        self._ids.clear()
        self._terms.clear()
        self._ordinals.clear()
        self._ordinals.shrink_to_fit()
        return vocab, ordinals
