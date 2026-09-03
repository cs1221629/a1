"""Small correctness checks for the submitted index and scorers."""
import math

import numpy as np
import pytest

from submission import bm25, boolean_vsm, indexer
from submission.indexer import InvertedIndex, tokenize


@pytest.fixture(autouse=True)
def _restore_default_tokenizer():
    """Tests that change the tokenizer must not leak that change into the
    next test — the setting is module-level global state."""
    yield
    indexer.configure_tokenizer()


def test_tokenizer_removes_stopwords_and_stems_without_topic_rules():
    """The original pinned tokenizer contract, kept as a regression check.

    This is the behaviour the first submission shipped (min token length 2).
    It is pinned explicitly rather than deleted so that a future change to
    stopwording or stemming still has to justify itself against a known
    reference, instead of the expectations silently following whatever the
    implementation happens to do.
    """
    indexer.configure_tokenizer(stoplist="minimal", stemmer="porter",
                                min_token_len=2)
    assert tokenize("What is a running experiment?") == ["run", "experi"]
    assert tokenize("SARS-CoV-2 infections") == ["sar", "cov", "infect"]


def test_shipped_tokenizer_keeps_single_character_tokens():
    """The shipped configuration keeps length-1 tokens.

    Chosen by scripts/tune.py --stage mintok, where it won all five
    cross-validation folds: "SARS-CoV-2" and "COVID-19" carry their
    trailing number as a separate token, and dropping it loses signal.
    """
    assert tokenize("SARS-CoV-2 infections") == ["sar", "cov", "2", "infect"]
    assert tokenize("What is a running experiment?") == ["run", "experi"]


def test_tokenizer_can_disable_stemming_and_use_a_larger_stoplist():
    indexer.configure_tokenizer(stoplist="standard", stemmer="none",
                                min_token_len=2)
    # "of"/"the" are in both stoplists; "running" is left unstemmed.
    assert tokenize("the running of experiments") == ["running", "experiments"]


def test_boolean_and_or_and_bm25_use_the_same_index():
    index = InvertedIndex()
    index.build([
        ("d1", "COVID treatments and treatments"),
        ("d2", "COVID testing"),
        ("d3", "unrelated document"),
    ])
    bm25.build(index)
    boolean_vsm.build(index)

    assert boolean_vsm.boolean_search("covid treatment", "and") == ["d1"]
    assert boolean_vsm.boolean_search("covid treatment", "or") == ["d1", "d2"]
    assert bm25.score("COVID treatment", 1)[0][0] == "d1"


# --- index structure ----------------------------------------------------

def _tiny_index():
    index = InvertedIndex()
    index.build([
        ("d1", "alpha beta beta"),
        ("d2", "alpha gamma"),
        ("d3", "delta"),
    ])
    return index


def test_index_records_exact_df_tf_and_doc_lengths():
    index = _tiny_index()
    # No stopword in this vocabulary, and each term stems to itself.
    assert index.N == 3
    assert index.doc_id_map == ["d1", "d2", "d3"]
    assert list(index.doc_len) == [3, 2, 1]
    assert index.avg_doc_len == pytest.approx(2.0)

    assert index.document_frequency("alpha") == 2
    assert index.document_frequency("beta") == 1
    assert index.document_frequency("nonexistent") == 0

    docs, tfs = index.postings_for("beta")
    assert list(docs) == [0] and list(tfs) == [2]      # tf=2 inside d1
    docs, tfs = index.postings_for("alpha")
    assert list(docs) == [0, 1] and list(tfs) == [1, 1]
    docs, tfs = index.postings_for("nonexistent")
    assert docs.size == 0 and tfs.size == 0


def test_postings_are_sorted_by_doc_id_within_each_term():
    """Scorers rely on this: the tie-break and the top-k selection both
    assume each term's posting block is doc-id ascending."""
    index = InvertedIndex()
    index.build([(f"d{i}", "shared term") for i in range(50)])
    docs, _ = index.postings_for("share")
    assert list(docs) == sorted(docs)
    assert len(set(docs.tolist())) == docs.size


# --- BM25 ---------------------------------------------------------------

def test_bm25_matches_hand_computed_scores():
    """Check the actual arithmetic, not just the resulting order."""
    index = _tiny_index()
    bm25.build(index)
    k1, b = 1.5, 0.75

    results = dict(bm25.score("alpha", 10, k1=k1, b=b))

    N, avgdl = 3, 2.0
    df = 2                                    # "alpha" appears in d1 and d2
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
    for doc_id, doc_len in (("d1", 3), ("d2", 2)):
        tf = 1
        K = k1 * (1 - b + b * doc_len / avgdl)
        expected = idf * (tf * (k1 + 1) / (tf + K))
        assert results[doc_id] == pytest.approx(expected, rel=1e-5)

    # d2 is shorter, so length normalisation ranks it above d1.
    assert results["d2"] > results["d1"]
    assert "d3" not in results


def test_bm25_delta_is_a_real_parameter():
    index = _tiny_index()
    bm25.build(index)
    plain = dict(bm25.score("alpha", 10, k1=1.5, b=0.75, delta=0.0))
    plus = dict(bm25.score("alpha", 10, k1=1.5, b=0.75, delta=0.5))
    df, N = 2, 3
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
    for doc_id in ("d1", "d2"):
        assert plus[doc_id] == pytest.approx(plain[doc_id] + 0.5 * idf, rel=1e-5)

    with pytest.raises(ValueError):
        bm25.score("alpha", 10, delta=-1.0)


def test_bm25_breaks_exact_ties_by_lowest_doc_id():
    """Two identical documents score identically; the lower doc id must win,
    including when the tie straddles the k-th position (a bare
    np.argpartition would drop an arbitrary one of them)."""
    index = InvertedIndex()
    index.build([(f"d{i}", "identical text here") for i in range(5)])
    bm25.build(index)

    full = bm25.score("identical", 5)
    assert [doc for doc, _ in full] == ["d0", "d1", "d2", "d3", "d4"]
    scores = [s for _, s in full]
    assert all(s == pytest.approx(scores[0]) for s in scores)

    truncated = bm25.score("identical", 2)
    assert [doc for doc, _ in truncated] == ["d0", "d1"]


def test_bm25_handles_empty_and_unknown_queries():
    index = _tiny_index()
    bm25.build(index)
    assert bm25.score("", 10) == []
    assert bm25.score("nonexistentterm", 10) == []
    assert bm25.score("alpha", 0) == []


def test_bm25_returns_at_most_k_results_sorted_descending():
    index = _tiny_index()
    bm25.build(index)
    results = bm25.score("alpha gamma", 1)
    assert len(results) == 1
    results = bm25.score("alpha gamma", 10)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# --- Boolean / VSM ------------------------------------------------------

def test_boolean_search_edge_cases():
    index = _tiny_index()
    boolean_vsm.build(index)

    assert boolean_vsm.boolean_search("nonexistent", "and") == []
    assert boolean_vsm.boolean_search("alpha nonexistent", "and") == []
    assert boolean_vsm.boolean_search("alpha nonexistent", "or") == ["d1", "d2"]
    # A repeated term must not change the result set.
    assert boolean_vsm.boolean_search("alpha alpha", "and") == ["d1", "d2"]
    assert boolean_vsm.boolean_search("", "and") == []
    assert boolean_vsm.boolean_search("alpha beta", "or") == ["d1", "d2"]
    assert boolean_vsm.boolean_search("alpha beta", "and") == ["d1"]

    with pytest.raises(ValueError):
        boolean_vsm.boolean_search("alpha", "xor")


def test_vsm_cosine_similarity_matches_hand_computation():
    index = _tiny_index()
    boolean_vsm.build(index)

    results = dict(boolean_vsm.vsm_score("beta", 10))
    # Only d1 contains "beta". Query vector is a single term, so cosine
    # similarity reduces to (w_d * w_q) / (||d|| * ||q||).
    N = 3
    idf_beta = math.log(N / 1)
    idf_alpha = math.log(N / 2)
    d1_norm = math.sqrt((1 * idf_alpha) ** 2 + (2 * idf_beta) ** 2)
    expected = (2 * idf_beta * idf_beta) / (d1_norm * idf_beta)
    assert results["d1"] == pytest.approx(expected, rel=1e-6)
    assert set(results) == {"d1"}


def test_vsm_edge_cases():
    index = _tiny_index()
    boolean_vsm.build(index)
    assert boolean_vsm.vsm_score("", 10) == []
    assert boolean_vsm.vsm_score("nonexistentterm", 10) == []
    assert boolean_vsm.vsm_score("alpha", 0) == []
    # Every returned similarity is a finite number in [0, 1].
    for _doc, sim in boolean_vsm.vsm_score("alpha beta", 10):
        assert 0.0 <= sim <= 1.0 + 1e-9
        assert not math.isnan(sim)


# --- persistence --------------------------------------------------------

def test_index_round_trips_through_save_and_load(tmp_path):
    index = _tiny_index()
    index.save(str(tmp_path))
    reloaded = InvertedIndex.load(str(tmp_path))

    assert reloaded.N == index.N
    assert reloaded.doc_id_map == index.doc_id_map
    assert reloaded.avg_doc_len == pytest.approx(index.avg_doc_len)
    assert np.array_equal(reloaded.doc_len, index.doc_len)
    assert np.array_equal(reloaded.post_doc, index.post_doc)
    assert np.array_equal(reloaded.post_tf, index.post_tf)
    assert np.array_equal(reloaded.post_off, index.post_off)
    assert reloaded.term_ids == index.term_ids

    bm25.build(index)
    before = bm25.score("alpha beta", 10)
    bm25.build(reloaded)
    after = bm25.score("alpha beta", 10)
    assert before == after


def test_load_restores_the_tokenizer_the_index_was_built_with(tmp_path):
    """An index built with one tokenizer and queried with another degrades
    silently, so the setting travels with the index."""
    indexer.configure_tokenizer(stoplist="standard", stemmer="none",
                                min_token_len=2)
    index = InvertedIndex()
    index.build([("d1", "running experiments")])
    index.save(str(tmp_path))

    indexer.configure_tokenizer()  # back to the shipped default
    InvertedIndex.load(str(tmp_path))
    assert indexer.tokenizer_config() == {
        "stoplist": "standard", "stemmer": "none", "min_token_len": 2,
        "compounds": False,
    }


# --- on-disk codec ------------------------------------------------------

def test_bucketed_encoding_round_trips_across_all_three_streams():
    from submission import codec
    values = np.array([0, 1, 254, 255, 256, 65534, 65535, 65536,
                       4_000_000_000], dtype=np.uint32)
    decoded = codec.decode_bucketed(*codec.encode_bucketed(values))
    assert np.array_equal(decoded, values)

    # Empty input, and input that never escapes, must both work.
    assert codec.decode_bucketed(*codec.encode_bucketed(
        np.zeros(0, dtype=np.uint32))).size == 0
    small = np.arange(200, dtype=np.uint32)
    assert np.array_equal(
        codec.decode_bucketed(*codec.encode_bucketed(small)), small)


def test_doc_gap_encoding_handles_doc_zero_as_a_block_start():
    """The classic off-by-one: with a `previous = 0` convention rather than
    `previous = -1`, a block whose first posting is doc 0 collides with one
    starting at doc 1. Pin the boundary explicitly."""
    from submission import codec
    post_off = np.array([0, 3, 5], dtype=np.int64)
    post_doc = np.array([0, 1, 7, 0, 9], dtype=np.int32)  # both blocks start at doc 0
    gaps = codec.encode_doc_gaps(post_doc, post_off)
    assert list(gaps) == [0, 0, 5, 0, 8]
    assert np.array_equal(codec.decode_doc_gaps(gaps, post_off), post_doc)


def test_doc_gap_encoding_round_trips_with_empty_blocks():
    from submission import codec
    post_off = np.array([0, 2, 2, 4], dtype=np.int64)   # middle term has no postings
    post_doc = np.array([3, 100, 0, 171331], dtype=np.int32)
    gaps = codec.encode_doc_gaps(post_doc, post_off)
    assert np.array_equal(codec.decode_doc_gaps(gaps, post_off), post_doc)


def test_front_coding_round_trips_including_long_and_unicode_terms():
    from submission import codec
    terms = sorted(["", "a", "ab", "abc", "abcd", "b", "covid", "covid19",
                    "x" * 300, "naive", "zzz"])
    assert codec.front_decode(codec.front_code(terms), len(terms)) == terms


def test_string_list_encoding_round_trips():
    from submission import codec
    ids = ["kgifmjvb", "wmfcey6f", "", "a-longer-document-identifier"]
    payload, l1, l2, l3 = codec.encode_strings(ids)
    assert codec.decode_strings(payload, l1, l2, l3) == ids


def test_string_list_encoding_survives_non_ascii_doc_ids():
    """Lengths are stored in bytes, so decoding must slice bytes too. If it
    decodes the whole payload first and slices the str, a single non-ASCII
    character corrupts that id and every id after it."""
    from submission import codec
    ids = ["ascii-id", "café-doc", "中文文档", "naïve", "plain"]
    payload, l1, l2, l3 = codec.encode_strings(ids)
    assert codec.decode_strings(payload, l1, l2, l3) == ids


def test_index_rejects_a_file_that_is_not_an_index(tmp_path):
    with open(tmp_path / "index.bin", "wb") as f:
        f.write(b"definitely not an index")
    with pytest.raises(ValueError, match="magic"):
        InvertedIndex.load(str(tmp_path))


# --- RM3 (implemented, measured, deliberately not shipped) ---------------

def test_rm3_forward_index_agrees_with_the_inverted_index():
    """rm3 rebuilds a document-major view at load time. If it disagreed with
    the postings it was derived from, the relevance model would be built
    from the wrong term vectors."""
    from submission import rm3
    index = InvertedIndex()
    index.build([("d1", "alpha beta beta"), ("d2", "alpha gamma"),
                 ("d3", "delta alpha")])
    rm3.build(index)
    terms = sorted(index.term_ids, key=index.term_ids.get)
    for doc in range(index.N):
        fwd_terms, fwd_tfs = rm3.terms_of(doc)
        assert list(fwd_terms) == sorted(fwd_terms), "terms must be ascending"
        for tid, tf in zip(fwd_terms, fwd_tfs):
            docs, tfs = index.postings_for(terms[int(tid)])
            j = list(docs).index(doc)
            assert tfs[j] == tf


def test_rm3_alpha_zero_reproduces_plain_bm25_ranking():
    """The reduction case: with no expansion weight, RM3 must rank exactly
    like BM25, or every comparison against it is against a moving target."""
    from submission import rm3
    index = InvertedIndex()
    index.build([("d1", "covid testing rapid"), ("d2", "covid vaccine"),
                 ("d3", "unrelated text"), ("d4", "rapid covid covid")])
    bm25.build(index)
    rm3.build(index)
    plain = [d for d, _ in bm25.score("covid rapid", 10, k1=1.5, b=0.6)]
    expanded_off = [d for d, _ in rm3.score("covid rapid", 10, alpha=0.0,
                                            k1=1.5, b=0.6)]
    assert plain == expanded_off

    with pytest.raises(ValueError):
        rm3.score("covid", 10, alpha=1.5)
