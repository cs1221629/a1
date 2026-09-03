"""Explain Boolean, VSM, and BM25 retrieval for one query on a corpus."""
import argparse
import math
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission import bm25, boolean_vsm
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex, tokenize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--corpus", default="data/toy/corpus.jsonl")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    texts = dict(corpus)
    index = InvertedIndex()
    index.build(corpus)
    bm25.build(index)
    boolean_vsm.build(index)
    terms = tokenize(args.query)
    print(f"Query: {args.query!r}\nTokens: {terms}\n")
    print(f"Collection: N={index.N}, average document length={index.avg_doc_len:.2f}\n")
    for term, qtf in Counter(terms).items():
        df = index.document_frequency(term)
        if not df:
            print(f"{term!r}: not in the index")
            continue
        idf = math.log((index.N - df + 0.5) / (df + 0.5) + 1.0)
        docs, tfs = index.postings_for(term)
        rows = [f"{index.doc_id_map[d]}(tf={tf}, len={index.doc_len[d]})"
                for d, tf in zip(docs, tfs)]
        print(f"{term!r}: qtf={qtf}, df={df}, BM25 idf={idf:.4f}")
        print("  postings: " + ", ".join(rows))

    print("\nBoolean AND:", boolean_vsm.boolean_search(args.query, "and"))
    print("Boolean OR: ", boolean_vsm.boolean_search(args.query, "or"))
    print("\nVSM ranking:")
    for doc_id, score in boolean_vsm.vsm_score(args.query, args.k):
        print(f"  {doc_id:>4}  {score:.4f}  {texts[doc_id]}")
    print(f"\nBM25 ranking (k1={args.k1}, b={args.b}):")
    for doc_id, score in bm25.score(args.query, args.k, args.k1, args.b):
        print(f"  {doc_id:>4}  {score:.4f}  {texts[doc_id]}")


if __name__ == "__main__":
    main()
