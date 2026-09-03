"""
scripts/tune.py — cross-validated parameter selection.

Supersedes scripts/tune_bm25.py, which reported the single best
configuration over all 50 dev topics. That is the classic way to overfit a
50-topic public set: with ~50 queries, differences of 0.005 nDCG between
neighbouring parameter settings are well inside sampling noise, and the
argmax reliably lands on a lucky peak that does not survive to held-out
topics. 50% of the assignment grade comes from private topics we never
see, so selection here is by 5-fold cross-validation over the dev topics
instead.

What "cross-validation" means in this context: we are not learning model
parameters from documents. We split the 50 *topics* (with their qrels)
into 5 fixed, seeded folds; for each candidate configuration we evaluate
each fold separately and report the mean across folds and the spread
between them. A configuration that wins on the mean while showing a small
spread is a broad plateau; one that wins the full-set argmax but swings
wildly across folds is a sharp peak we should not trust.

Stages (each gated on the previous stage's winner, so the search stays
linear rather than combinatorial):

    --stage tokenizer   3-way tokenizer comparison at fixed BM25 params
    --stage mintok      keep single-character tokens or not
    --stage bm25        k1 / b / delta grid on the winning tokenizer
    --stage blend       BM25 + VSM blend weight lambda

Usage:
    python scripts/tune.py --stage tokenizer
    python scripts/tune.py --stage bm25 --stoplist standard
"""
import argparse
import itertools
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from harness.metrics import ndcg_at_k, average_precision  # noqa: E402
from harness.trec_io import read_qrels, read_queries  # noqa: E402
from submission import bm25, boolean_vsm, custom_scorer, indexer  # noqa: E402
from submission.corpus_utils import iter_corpus  # noqa: E402
from submission.indexer import InvertedIndex  # noqa: E402

CORPUS = "data/full/corpus.jsonl"
QUERIES = "data/full/queries_dev.tsv"
QRELS = "data/full/qrels_dev.txt"
N_FOLDS = 5
SEED = 20260831  # fixed so folds are identical across runs and stages


def make_folds(qids, n_folds=N_FOLDS, seed=SEED):
    """Deterministic assignment of topics to folds."""
    rng = np.random.RandomState(seed)
    shuffled = list(qids)
    rng.shuffle(shuffled)
    return [shuffled[i::n_folds] for i in range(n_folds)]


def build_index(stoplist, stemmer, min_token_len, compounds=False):
    indexer.configure_tokenizer(stoplist=stoplist, stemmer=stemmer,
                                min_token_len=min_token_len,
                                compounds=compounds)
    t0 = time.perf_counter()
    index = InvertedIndex()
    index.build(iter_corpus(CORPUS))
    build_s = time.perf_counter() - t0
    bm25.build(index)
    boolean_vsm.build(index)
    return index, build_s


def evaluate(scorer, queries, qrels, folds):
    """Score every topic once and return per-topic nDCG plus aggregates.

    Note on what the folds can and cannot tell us: because we do not train
    anything on a fold, the mean of equal-sized fold means is *identically*
    the mean over all topics. Reporting a "CV mean" per configuration would
    therefore just restate full-dev nDCG. The folds earn their keep two
    other ways, both used below: the spread across folds measures how
    stable a configuration is, and `cv_select()` uses fold-separated
    selection to estimate how well the *selection procedure itself*
    generalizes to topics it did not choose on.
    """
    per_q_ndcg, per_q_ap, latencies = {}, {}, []
    for qid, text in queries:
        if qid not in qrels:
            continue
        t0 = time.perf_counter()
        results = scorer(text, 10)
        latencies.append(time.perf_counter() - t0)
        ranked = [doc_id for doc_id, _ in results]
        per_q_ndcg[qid] = ndcg_at_k(ranked, qrels[qid], k=10)
        per_q_ap[qid] = average_precision(ranked, qrels[qid])

    fold_means = [
        sum(per_q_ndcg[q] for q in fold if q in per_q_ndcg) /
        max(1, len([q for q in fold if q in per_q_ndcg]))
        for fold in folds
    ]
    return {
        "per_query_ndcg": per_q_ndcg,
        "full_ndcg": sum(per_q_ndcg.values()) / len(per_q_ndcg),
        "full_map": sum(per_q_ap.values()) / len(per_q_ap),
        "fold_means": fold_means,
        "spread": max(fold_means) - min(fold_means) if fold_means else 0.0,
        "latency": statistics.mean(latencies),
    }


def _mean_over(rows_per_query, qids):
    vals = [rows_per_query[q] for q in qids if q in rows_per_query]
    return sum(vals) / len(vals) if vals else 0.0


def cv_select(rows, folds):
    """Fold-separated selection: for each fold, choose the configuration
    that wins on the other four folds, then score that choice on the
    held-out fold it never saw.

    The average of those held-out scores is an honest estimate of what this
    whole tuning procedure is worth on unseen topics — which is the number
    that matters, since half the grade rides on private topics. It is
    usually a little below the full-dev argmax, and the gap between them is
    exactly the overfitting we are trying not to pay for.

    Also reports how often each configuration was the per-fold winner:
    a configuration chosen by all five folds is a broad plateau, one chosen
    once is a fold-specific artifact.
    """
    held_out_scores, chosen = [], []
    for i, fold in enumerate(folds):
        train_qids = [q for j, f in enumerate(folds) if j != i for q in f]
        winner = max(rows, key=lambda r: _mean_over(r["per_query_ndcg"], train_qids))
        held_out_scores.append(_mean_over(winner["per_query_ndcg"], fold))
        chosen.append(winner["label"])

    return {
        "cv_estimate": statistics.mean(held_out_scores),
        "per_fold_heldout": held_out_scores,
        "per_fold_choice": chosen,
        "choice_counts": {lab: chosen.count(lab) for lab in set(chosen)},
    }


def report(rows, folds):
    """Print a leaderboard of candidate configurations, plus the honest
    fold-separated estimate for the selection procedure."""
    rows = sorted(rows, key=lambda r: -r["full_ndcg"])
    width = max(len(r["label"]) for r in rows)
    print(f"\n{'config'.ljust(width)}  {'nDCG@10':>8} {'spread':>7} "
          f"{'MAP':>7} {'lat ms':>7}")
    print("-" * (width + 34))
    for r in rows:
        print(f"{r['label'].ljust(width)}  {r['full_ndcg']:8.4f} {r['spread']:7.4f} "
              f"{r['full_map']:7.4f} {1000 * r['latency']:7.2f}")

    best = rows[0]
    cv = cv_select(rows, folds)
    print(f"\nfull-dev argmax:  {best['label']}  nDCG@10 {best['full_ndcg']:.4f}")
    print(f"fold-separated CV estimate: {cv['cv_estimate']:.4f}  "
          f"(optimism {best['full_ndcg'] - cv['cv_estimate']:+.4f})")
    print(f"per-fold held-out: {[round(s, 4) for s in cv['per_fold_heldout']]}")
    print("per-fold winner counts:")
    for lab, n in sorted(cv["choice_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {n}/{len(folds)}  {lab}")
    return rows, cv


def save(rows, cv, name):
    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", f"tune_{name}.json")
    serializable = [{k: v for k, v in r.items() if k != "per_query_ndcg"}
                    for r in rows]
    with open(path, "w") as f:
        json.dump({"configs": serializable, "cross_validation": cv,
                   "seed": SEED, "n_folds": N_FOLDS}, f, indent=2)
    print(f"wrote {path}")


def stage_tokenizer(queries, qrels, folds, args):
    """Round 1: compare tokenization schemes at fixed BM25 parameters."""
    configs = [
        ("A: minimal stoplist + porter", "minimal", "porter", 2),
        ("B: standard stoplist + porter", "standard", "porter", 2),
        ("C: minimal stoplist, no stemming", "minimal", "none", 2),
    ]
    rows = []
    for label, stoplist, stemmer, min_len in configs:
        index, build_s = build_index(stoplist, stemmer, min_len)
        res = evaluate(lambda q, k: bm25.score(q, k, args.k1, args.b, args.delta),
                       queries, qrels, folds)
        res.update({
            "label": label, "stoplist": stoplist, "stemmer": stemmer,
            "min_token_len": min_len, "build_seconds": build_s,
            "num_terms": len(index.term_ids),
            "num_postings": int(index.post_doc.size),
        })
        rows.append(res)
        print(f"  {label}: nDCG {res['full_ndcg']:.4f}  "
              f"terms {len(index.term_ids):,}  "
              f"postings {index.post_doc.size:,}  build {build_s:.1f}s")
    save(*report(rows, folds), "tokenizer")


def stage_mintok(queries, qrels, folds, args):
    """Follow-up: does keeping single-character tokens help?

    "SARS-CoV-2" tokenizes to sars/cov/2 and "COVID-19" to covid/19; the
    shipped min length of 2 keeps "19" but drops "2". Whether that final
    character carries retrieval signal is measured here, not assumed.
    """
    rows = []
    for min_len in (1, 2):
        index, build_s = build_index(args.stoplist, args.stemmer, min_len)
        res = evaluate(lambda q, k: bm25.score(q, k, args.k1, args.b, args.delta),
                       queries, qrels, folds)
        res.update({
            "label": f"min_token_len={min_len}", "stoplist": args.stoplist,
            "stemmer": args.stemmer, "min_token_len": min_len,
            "build_seconds": build_s,
            "num_postings": int(index.post_doc.size),
        })
        rows.append(res)
        print(f"  min_token_len={min_len}: nDCG {res['full_ndcg']:.4f}  "
              f"postings {index.post_doc.size:,}")
    save(*report(rows, folds), "mintok")


def stage_tokenizer2(queries, qrels, folds, args):
    """Round 2 of the tokenizer comparison, on the two axes round 1 never
    tested: how aggressive the stemmer should be, and whether hyphenated
    compounds should also be indexed glued.

    Round 1 established that stemming matters a lot here (dropping Porter
    cost 0.0425 nDCG@10) but only compared "Porter" against "nothing".
    Porter is derivational and this is a scientific collection, so the
    interesting question is the middle option: strip inflections only.

    Evaluated against the *shipped* scorer (the BM25+VSM blend), not plain
    BM25, because that is what actually ranks — a tokenizer that helps BM25
    alone but not the blend is not an improvement to this submission.
    """
    configs = [
        ("porter, no compounds  (shipped)", args.stoplist, "porter", 1, False),
        ("porter, + compounds", args.stoplist, "porter", 1, True),
        ("light, no compounds", args.stoplist, "light", 1, False),
        ("light, + compounds", args.stoplist, "light", 1, True),
        ("none, + compounds", args.stoplist, "none", 1, True),
    ]
    rows = []
    for label, stoplist, stemmer, min_len, compounds in configs:
        index, build_s = build_index(stoplist, stemmer, min_len, compounds)
        custom_scorer.build(index)
        res = evaluate(
            lambda q, k: custom_scorer.score(q, k, lam=args.lam, k1=args.k1,
                                             b=args.b, delta=args.delta),
            queries, qrels, folds)
        res.update({
            "label": label, "stoplist": stoplist, "stemmer": stemmer,
            "min_token_len": min_len, "compounds": compounds,
            "build_seconds": build_s,
            "num_terms": len(index.term_ids),
            "num_postings": int(index.post_doc.size),
        })
        rows.append(res)
        print(f"  {label}: nDCG {res['full_ndcg']:.4f}  spread {res['spread']:.4f}  "
              f"terms {len(index.term_ids):,}  postings {index.post_doc.size:,}  "
              f"build {build_s:.1f}s")
    save(*report(rows, folds), "tokenizer2")


def stage_bm25(queries, qrels, folds, args):
    """Round 2: k1 / b / delta grid on the chosen tokenizer."""
    build_index(args.stoplist, args.stemmer, args.min_token_len)
    k1_values = [0.6, 0.9, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0]
    b_values = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.85, 1.0]
    delta_values = [0.0, 0.25, 0.5, 1.0]

    rows = []
    total = len(k1_values) * len(b_values) * len(delta_values)
    for n, (k1, b, delta) in enumerate(
            itertools.product(k1_values, b_values, delta_values), 1):
        res = evaluate(lambda q, k: bm25.score(q, k, k1, b, delta),
                       queries, qrels, folds)
        res.update({"label": f"k1={k1} b={b} d={delta}", "k1": k1, "b": b,
                    "delta": delta})
        rows.append(res)
        if n % 50 == 0 or n == total:
            print(f"  {n}/{total} configs evaluated")
    rows, cv = report(rows, folds)
    save(rows, cv, "bm25")

    write_surface_csv(rows)


def write_surface_csv(rows, delta=0.0, path="runs/tune_bm25_surface.csv"):
    """k1 x b surface at one delta — the report's required sweep plot.

    Written at delta=0 by default rather than at the grid's argmax delta:
    delta's marginal effect is monotonically negative (mean nDCG@10 falls
    0.6110 -> 0.5864 as delta goes 0 -> 1), so the plain-BM25 plane is the
    one worth plotting.
    """
    subset = [r for r in rows if r["delta"] == delta]
    with open(path, "w") as f:
        f.write("k1,b,delta,ndcg10,fold_spread\n")
        for r in sorted(subset, key=lambda r: (r["k1"], r["b"])):
            f.write(f"{r['k1']},{r['b']},{r['delta']},"
                    f"{r['full_ndcg']:.6f},{r['spread']:.6f}\n")
    print(f"wrote {path} ({len(subset)} rows at delta={delta})")


def stage_finalists(queries, qrels, folds, args):
    """Head-to-head on the handful of configurations the grid left standing.

    The grid's full-dev argmax is not trustworthy on its own: `cv_select`
    measured +0.017 of optimism from picking an argmax over 440 candidates
    on 50 topics, which is larger than the entire spread between the top
    configurations. So instead of ranking by a single number, this prints
    each finalist's per-fold nDCG and counts pairwise fold wins against the
    incumbent, which shows whether a difference is consistent or is one or
    two lucky topics in one fold.
    """
    finalists = [
        (2.0, 0.6, 0.0),   # incumbent: current shipped parameters
        (2.5, 0.5, 0.0),   # centre of the 3x3-smoothed plateau
        (2.2, 0.6, 0.0),
        (2.5, 0.6, 0.0),
        (3.0, 0.4, 0.0),
        (3.5, 0.6, 0.5),   # full-dev argmax from the grid
    ]
    build_index(args.stoplist, args.stemmer, args.min_token_len)

    rows = []
    for k1, b, delta in finalists:
        res = evaluate(lambda q, k: bm25.score(q, k, k1, b, delta),
                       queries, qrels, folds)
        res.update({"label": f"k1={k1} b={b} d={delta}", "k1": k1, "b": b,
                    "delta": delta})
        rows.append(res)

    width = max(len(r["label"]) for r in rows)
    print(f"\n{'config'.ljust(width)}  {'full':>7}  per-fold nDCG@10")
    print("-" * (width + 50))
    for r in rows:
        folds_str = " ".join(f"{m:.4f}" for m in r["fold_means"])
        print(f"{r['label'].ljust(width)}  {r['full_ndcg']:7.4f}  {folds_str}")

    incumbent = rows[0]
    print(f"\nper-fold record vs. incumbent ({incumbent['label']}):")
    for r in rows[1:]:
        wins = sum(1 for a, bb in zip(r["fold_means"], incumbent["fold_means"])
                   if a > bb)
        mean_delta = statistics.mean(
            a - bb for a, bb in zip(r["fold_means"], incumbent["fold_means"]))
        print(f"  {r['label'].ljust(width)}  wins {wins}/{len(folds)} folds, "
              f"mean delta {mean_delta:+.4f}")

    save(rows, cv_select(rows, folds), "finalists")


def stage_blend(queries, qrels, folds, args):
    """BM25 + VSM blend weight. lambda=1.0 is plain BM25."""
    build_index(args.stoplist, args.stemmer, args.min_token_len)
    custom_scorer.build(bm25._INDEX)
    rows = []
    for lam in [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        res = evaluate(
            lambda q, k: custom_scorer.score(
                q, k, lam=lam, k1=args.k1, b=args.b, delta=args.delta),
            queries, qrels, folds)
        res.update({"label": f"lambda={lam}", "lam": lam})
        rows.append(res)
        print(f"  lambda={lam}: nDCG {res['full_ndcg']:.4f}  "
              f"lat {1000 * res['latency']:.2f}ms")
    save(*report(rows, folds), "blend")


STAGES = {
    "tokenizer": stage_tokenizer,
    "tokenizer2": stage_tokenizer2,
    "mintok": stage_mintok,
    "bm25": stage_bm25,
    "finalists": stage_finalists,
    "blend": stage_blend,
}


def main():
    parser = argparse.ArgumentParser(description="Cross-validated tuning.")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--stoplist", default=indexer.DEFAULT_STOPLIST)
    parser.add_argument("--stemmer", default=indexer.DEFAULT_STEMMER)
    parser.add_argument("--min-token-len", type=int,
                        default=indexer.DEFAULT_MIN_TOKEN_LEN)
    parser.add_argument("--k1", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=0.6)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=0.8)
    args = parser.parse_args()

    queries = read_queries(QUERIES)
    qrels = read_qrels(QRELS)
    qids = [q for q, _ in queries if q in qrels]
    folds = make_folds(qids)
    print(f"{len(qids)} topics, {N_FOLDS} folds "
          f"(sizes {[len(f) for f in folds]}), seed {SEED}")

    STAGES[args.stage](queries, qrels, folds, args)


if __name__ == "__main__":
    main()
