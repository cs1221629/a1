"""
tests/test_fast_tokenizer.py — the optional C++ extension must produce
exactly the same index as the pure-Python path.

submission/indexer.py picks `_stream_fast` over `_stream_python` whenever
the extension is importable, so the two are interchangeable by
construction only if they genuinely agree. That is the property worth
pinning: a divergence here would not crash, it would silently change
retrieval on the graded corpus.

Every test skips cleanly when the extension is not built, so the suite
stays green on a machine with no compiler (which is the same situation the
fallback exists for).
"""
import numpy as np
import pytest

from submission import indexer
from submission.indexer import InvertedIndex

pytestmark = pytest.mark.skipif(
    indexer._fast_tokenizer is None,
    reason="C++ tokenizer extension not built (submission/setup.py)",
)


DOCS = [
    ("d1", "SARS-CoV-2 transmission in humans"),
    ("d2", "the RNA virus replicates; RNA RNA"),
    ("d3", ""),                                   # empty document
    ("d4", "   ...   "),                          # no alphanumeric content
    ("d5", "COVID19 covid19 CoViD19 mixed-case"),
    ("d6", "café naïve 中文 text"),   # non-ASCII
    ("d7", "a b c 1 2 3"),                        # single-character tokens
    ("d8", "trailing token at end of buffer"),
    ("d9", "İSTANBUL conclusİons"),     # dotted capital I
    ("d10", "x" * 300 + " " + "y" * 5),           # very long token
]


def _streams(docs):
    """Run both streaming passes over the same documents."""
    py_ids, py_lens = [], []
    py_terms, py_raw = indexer._stream_python(iter(docs), py_ids, py_lens)

    fast_ids, fast_lens = [], []
    fast_terms, fast_raw = indexer._stream_fast(iter(docs), fast_ids, fast_lens)
    return (py_ids, py_lens, py_terms, py_raw), (fast_ids, fast_lens, fast_terms, fast_raw)


def test_streaming_pass_matches_python():
    (py_ids, py_lens, py_terms, py_raw), (f_ids, f_lens, f_terms, f_raw) = _streams(DOCS)

    assert py_ids == f_ids
    assert py_lens == f_lens
    # Same vocabulary, and the same ordinal for every type. First-seen
    # numbering happens to agree too, which makes the ordinal streams
    # directly comparable.
    assert py_terms == f_terms
    assert np.array_equal(py_raw, f_raw)


def test_token_stream_matches_the_reference_tokenizer():
    """The C++ scan must agree with split_bytes(), which is the definition
    of the tokenizer everywhere else (including at query time)."""
    _, (_, f_lens, f_terms, f_raw) = _streams(DOCS)

    expected = [tok for _, text in DOCS for tok in indexer.split_bytes(text)]
    actual = [f_terms[i] for i in f_raw]
    assert actual == expected
    assert f_lens == [len(indexer.split_bytes(text)) for _, text in DOCS]


def test_full_index_is_identical_either_way(monkeypatch):
    """End-to-end: the same corpus indexed both ways, compared array by
    array. This is the property that lets the extension ship."""
    fast = InvertedIndex()
    fast.build(iter(DOCS))

    monkeypatch.setattr(indexer, "_fast_tokenizer", None)
    slow = InvertedIndex()
    slow.build(iter(DOCS))

    assert fast.term_ids == slow.term_ids
    assert fast.doc_id_map == slow.doc_id_map
    assert fast.N == slow.N
    assert fast.avg_doc_len == pytest.approx(slow.avg_doc_len, rel=0, abs=1e-12)
    for field in ("post_off", "post_doc", "post_tf", "doc_len"):
        assert np.array_equal(getattr(fast, field), getattr(slow, field)), field


def test_compound_tokenizer_falls_back_to_python():
    """The extension cannot express the compound option, so build() must
    route around it rather than silently indexing the wrong tokens."""
    indexer.configure_tokenizer(compounds=True)
    try:
        index = InvertedIndex()
        index.build([("d1", "sars-cov-2 study")])
        assert "sarscov2" in index.term_ids
    finally:
        indexer.configure_tokenizer()
