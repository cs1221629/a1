"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
        Called once, in its own process, with the path to a corpus.jsonl
        file (see data/README.md) and a directory to write your index
        into. Build whatever index and statistics you need, and WRITE
        THEM TO index_dir. The harness runs build_index() and
        load_index()/retrieve() in two SEPARATE processes on purpose (see
        harness/run_harness.py's module docstring) — nothing you only
        hold in memory here survives into load_index(). This call is
        timed as your "index build time" efficiency metric. The harness
        also measures the on-disk byte size of index_dir once this
        returns — that's your "index size" score (assignment Section 7),
        so write only what retrieve() actually needs, and consider
        compressing it.

    load_index(index_dir: str) -> None
        Called once, in a fresh process, before any retrieve() calls.
        Reconstruct everything retrieve() needs by reading index_dir —
        and only index_dir; there is no leftover state from
        build_index() to fall back on. Timed as your "index load time".

    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]
        Called once per query, only after load_index() has run in the
        same process. Return up to k (doc_id, score) pairs, sorted by
        score descending (highest score = most relevant). This is exactly
        the ranking the harness scores with nDCG@10 / MAP@10. doc_id values
        must be ones that appeared in the corpus passed to build_index().

This file ships with a trivial, fully-working baseline — return the first
k documents in the order build_index() saw them, ignoring the query
entirely — wired up below. It actually persists to disk and reloads
correctly, so it exercises the full build -> disk -> fresh process -> load
-> query path end-to-end from your very first commit. Its scores will be
close to zero; replace the logic, but keep the same
persist-in-build / reconstruct-in-load shape.
"""
import json
import os
from typing import List, Optional, Tuple

from submission.corpus_utils import iter_corpus
from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex

_INDEX: Optional[InvertedIndex] = None

# Selected by scripts/tune_bm25.py on the released development topics.
# Keep these at module scope so the final entry's choice is explicit.
BM25_K1 = 2.0
BM25_B = 0.6
BM25_DELTA = 0.0

def build_index(corpus_path: str, index_dir: str) -> None:
    """Load the corpus, build whatever index structures you need, and
    write everything retrieve() will need into `index_dir`.
    """
    index = InvertedIndex()
    index.build(iter_corpus(corpus_path))
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct everything retrieve() needs, reading only from
    `index_dir`. Runs once, in a fresh process, before any retrieve()
    calls — there is no leftover state from build_index() to rely on.
    """
    global _INDEX
    _INDEX = InvertedIndex.load(index_dir)
    bm25.build(_INDEX)
    bm25.configure(BM25_K1, BM25_B)
    boolean_vsm.build(_INDEX)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, best first."""
    if _INDEX is None:
        raise RuntimeError(
            "retrieve() called before load_index(); the harness always "
            "calls build_index(corpus_path, index_dir) and then "
            "load_index(index_dir) — in that order, in two separate "
            "processes — before any retrieve() calls. If you're testing "
            "manually, do the same."
        )

    return bm25.score(query, k, k1=BM25_K1, b=BM25_B, delta=BM25_DELTA)
