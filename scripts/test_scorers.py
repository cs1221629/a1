from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex
from submission import boolean_vsm, bm25

corpus = load_corpus("data/toy/corpus.jsonl")
idx = InvertedIndex()
idx.build(corpus)
idx.save("runs/temp_index")

idx2 = InvertedIndex.load("runs/temp_index")
boolean_vsm.build(idx2)
bm25.build(idx2)

print("Boolean AND 'coffee', 'caffeine':", boolean_vsm.boolean_search("coffee caffeine", "and"))
print("Boolean OR 'coffee', 'caffeine':", boolean_vsm.boolean_search("coffee caffeine", "or"))

print("VSM 'coffee caffeine':", boolean_vsm.vsm_score("coffee caffeine", 5))
print("BM25 'coffee caffeine':", bm25.score("coffee caffeine", 5))
