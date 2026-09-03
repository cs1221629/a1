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
from submission import bm25, boolean_vsm, custom_scorer
from submission.indexer import InvertedIndex

_INDEX: Optional[InvertedIndex] = None

# All four constants below were selected by scripts/tune.py using
# fold-separated cross-validation over the 50 development topics, never by
# taking the argmax over all 50 — with only 50 topics the argmax over a
# large grid is worth about +0.017 nDCG of pure optimism (measured; see
# runs/tune_bm25.json), which is more than the spread between the top
# configurations. Keep them at module scope so the entry's choices are
# explicit and reproducible.

# k1 was retuned *after* the blend below was added, which matters: a 440-point
# k1/b/delta grid over plain BM25 had preferred 2.0, but the blend shifts the
# optimum. All five folds independently picked 1.5 without seeing their
# held-out topics, for an honest fold-separated gain of +0.0054 nDCG@10 at
# exactly 0.0000 optimism -- the cleanest signal any sweep in this project
# produced. b and lambda were then re-swept at k1=1.5 and both were already
# optimal: selecting any other value scored *worse* on held-out folds.
BM25_K1 = 1.5
BM25_B = 0.6
BM25_DELTA = 0.0

# Blend weight for custom_scorer: score = lam*norm(BM25) + (1-lam)*norm(cosine).
# lam=1.0 is exactly plain BM25. Every value in [0.7, 0.9] beat plain BM25
# on 4 of 5 folds (runs/tune_blend.json); 0.8 is the centre of that
# plateau, worth about +0.014 nDCG@10 over BM25 alone.
BLEND_LAMBDA = 0.8

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
    custom_scorer.build(_INDEX)

    # Warm up before the harness starts timing queries. The harness issues
    # no warm-up query and averages every latency it measures, so the first
    # query would otherwise pay for lazily-initialised numpy machinery and
    # the Porter stemmer's caches.
    retrieve("covid-19 transmission", 10)


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

    return custom_scorer.score(query, k, lam=BLEND_LAMBDA, k1=BM25_K1,
                               b=BM25_B, delta=BM25_DELTA)
