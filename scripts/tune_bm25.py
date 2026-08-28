"""Safely sweep BM25 parameters against a local dev set."""
import argparse
import itertools
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.metrics import evaluate_run
from harness.trec_io import read_queries, read_qrels
from submission import bm25
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/toy/corpus.jsonl")
    parser.add_argument("--queries", default="data/toy/queries_dev.tsv")
    parser.add_argument("--qrels", default="data/toy/qrels_dev.txt")
    parser.add_argument("--out", default="runs/bm25_tuning.json")
    parser.add_argument(
        "--k1-values", type=float, nargs="+",
        default=[0.6, 0.9, 1.2, 1.5, 1.8, 2.1],
        help="One or more k1 values to evaluate.",
    )
    parser.add_argument(
        "--b-values", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="One or more b values to evaluate.",
    )
    args = parser.parse_args()

    index = InvertedIndex()
    index.build(load_corpus(args.corpus))
    bm25.build(index)
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    results = []
    for k1, b in itertools.product(args.k1_values, args.b_values):
        run = {qid: bm25.score(query, 10, k1=k1, b=b) for qid, query in queries}
        metrics = evaluate_run(run, qrels, k=10)["aggregate"]
        results.append({"k1": k1, "b": b, **metrics})

    results.sort(key=lambda row: (-row["ndcg@10"], -row["map@10"], row["k1"], row["b"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Top settings (ranked by nDCG@10, then MAP@10):")
    for row in results[:10]:
        print(f"k1={row['k1']:.1f}, b={row['b']:.2f}: "
              f"nDCG@10={row['ndcg@10']:.4f}, MAP@10={row['map@10']:.4f}")
    print(f"Full sweep written to {args.out}")


if __name__ == "__main__":
    main()
