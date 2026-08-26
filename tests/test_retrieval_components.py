"""Small correctness checks for the submitted index and scorers."""
from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex, tokenize


def test_tokenizer_removes_stopwords_and_stems_without_topic_rules():
    assert tokenize("What is a running experiment?") == ["run", "experi"]
    assert tokenize("SARS-CoV-2 infections") == ["sar", "cov", "infect"]


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
