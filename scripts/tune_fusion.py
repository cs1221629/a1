"""
scripts/tune_fusion.py — compare ways of fusing the three ranking signals.

Why this is a separate script from scripts/tune.py: tune.py rebuilds the
index once per *tokenizer* configuration, which is the right shape for
tokenizer and BM25 parameter sweeps. Fusion sweeps do not touch the index
at all — every configuration re-ranks the same three score vectors — so
this script computes the raw per-candidate scores for BM25, cosine VSM and
the Dirichlet LM exactly once per topic, caches them, and then evaluates
hundreds of fusion settings against the cache. That turns a multi-hour
sweep into a couple of minutes.

Two families are compared:

  minmax  s = w_bm*n(bm25) + w_vsm*n(cos) + w_lm*n(lm),  n = per-query min-max
  rrf     s = sum_i w_i / (K + rank_i(d)),  ranks taken within each signal

RRF is the more interesting candidate here. The competition round is
scored on a *different collection* from data/full (see the plan file), and
min-max normalisation is sensitive to a signal's score distribution, which
is exactly what changes when the collection changes. RRF only reads the
order, so it transfers better — that is the reason to prefer it even at
equal measured nDCG.

Selection discipline matches scripts/tune.py: report the full-dev number
*and* the five fold means, and prefer a configuration that wins broadly
over one that wins big on the mean.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import ndcg_at_k, average_precision
from submission import bm25, boolean_vsm, lm_dirichlet
from submission.corpus_utils import iter_corpus
from submission.indexer import InvertedIndex, tokenize

CORPUS = "data/full/corpus.jsonl"
QUERIES = "data/full/queries_dev.tsv"
QRELS = "data/full/qrels_dev.txt"
N_FOLDS = 5
SEED = 20240501


def load_queries():
    out = []
    with open(QUERIES, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            qid, text = line.split("\t", 1)
            out.append((qid, text))
    return out


def load_qrels():
    qrels = {}
    with open(QRELS, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            if int(parts[3]) > 0:
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return qrels


def make_folds(qids):
    rng = np.random.RandomState(SEED)
    shuffled = list(qids)
    rng.shuffle(shuffled)
    return [shuffled[i::N_FOLDS] for i in range(N_FOLDS)]


def raw_signals(index, query, k1, b, mu):
    """Return (candidates, bm25, cosine, lm) over one shared candidate set.

    All three scorers accumulate over the union of the postings of the
    query terms the index knows, so the candidate set is identical for all
    three by construction; computing it once keeps the three vectors
    aligned index-for-index.
    """
    bm25.configure(k1, b)
    lm_dirichlet.configure(mu)

    tokens = tokenize(query)
    q_tf = {}
    for token in tokens:
        q_tf[token] = q_tf.get(token, 0) + 1

    acc_bm = np.zeros(index.N, dtype=np.float64)
    acc_vsm = np.zeros(index.N, dtype=np.float64)
    acc_lm = np.zeros(index.N, dtype=np.float64)
    mask = np.zeros(index.N, dtype=np.uint8)

    q_norm_sq = 0.0
    matched_qlen = 0
    touched = False
    for term, tf_q in q_tf.items():
        tid = index.term_ids.get(term)
        if tid is None:
            continue
        lo = int(index.post_off[tid])
        hi = int(index.post_off[tid + 1])
        if hi <= lo:
            continue
        docs = index.post_doc[lo:hi]
        tfs = index.post_tf[lo:hi].astype(np.float64)

        acc_bm[docs] += tf_q * float(bm25._IDF[tid]) * bm25._WEIGHT[lo:hi]

        idf_vsm = float(boolean_vsm._IDF[tid])
        q_w = tf_q * idf_vsm
        q_norm_sq += q_w * q_w
        acc_vsm[docs] += q_w * (tfs * idf_vsm)

        acc_lm[docs] += tf_q * lm_dirichlet._WEIGHT[lo:hi]

        mask[docs] = 1
        matched_qlen += tf_q
        touched = True

    if not touched:
        empty = np.zeros(0)
        return np.zeros(0, dtype=np.int64), empty, empty, empty

    cands = np.flatnonzero(mask)
    norms = boolean_vsm._DOC_NORMS[cands]
    q_norm = np.sqrt(q_norm_sq)
    cos = np.where(norms > 0.0, acc_vsm[cands] / (q_norm * np.maximum(norms, 1e-12)), 0.0)
    lm = acc_lm[cands] + matched_qlen * lm_dirichlet._LEN_NORM[cands]
    return cands, acc_bm[cands], cos, lm


def minmax(v):
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def rank_reciprocal(v, K, depth):
    """1/(K + rank) for the top `depth` of v, 0 for everything else.

    Truncating at a depth is what makes RRF cheap: a full argsort of every
    candidate costs far more than the fusion is worth, and documents
    ranked below ~1000 by every signal cannot reach a top-10 that any
    signal already fills.
    """
    out = np.zeros_like(v)
    n = min(depth, v.size)
    if n == 0:
        return out
    if n < v.size:
        sel = np.argpartition(-v, n - 1)[:n]
    else:
        sel = np.arange(v.size)
    order = sel[np.argsort(-v[sel], kind="stable")]
    out[order] = 1.0 / (K + np.arange(1, n + 1))
    return out


def top_ids(cands, scores, index, k=10):
    n = min(k, cands.size)
    if n == 0:
        return []
    sel = np.argpartition(-scores, n - 1)[:n]
    sel = sel[np.lexsort((cands[sel], -scores[sel]))]
    return [index.doc_id_map[int(c)] for c in cands[sel]]


def evaluate(fuse, cache, qrels, folds, index):
    per_q, per_ap = {}, {}
    for qid, (cands, bm, cos, lm) in cache.items():
        if cands.size == 0:
            per_q[qid] = 0.0
            per_ap[qid] = 0.0
            continue
        ranked = top_ids(cands, fuse(bm, cos, lm), index)
        per_q[qid] = ndcg_at_k(ranked, qrels[qid], k=10)
        per_ap[qid] = average_precision(ranked, qrels[qid])
    fold_means = [
        sum(per_q[q] for q in fold if q in per_q) / max(1, len([q for q in fold if q in per_q]))
        for fold in folds
    ]
    return {
        "full_ndcg": sum(per_q.values()) / len(per_q),
        "full_map": sum(per_ap.values()) / len(per_ap),
        "fold_means": fold_means,
        "spread": max(fold_means) - min(fold_means),
        "per_query_ndcg": per_q,
    }


def cv_select(rows, folds):
    """Fold-separated selection, same contract as scripts/tune.py:
    pick the winner on four folds, score it on the fifth it never saw."""
    held = []
    picks = []
    for i, fold in enumerate(folds):
        train = [q for j, f in enumerate(folds) if j != i for q in f]
        best, best_score = None, -1.0
        for row in rows:
            pq = row["per_query_ndcg"]
            vals = [pq[q] for q in train if q in pq]
            m = sum(vals) / len(vals) if vals else 0.0
            if m > best_score:
                best, best_score = row, m
        pq = best["per_query_ndcg"]
        vals = [pq[q] for q in fold if q in pq]
        held.append(sum(vals) / len(vals) if vals else 0.0)
        picks.append(best["label"])
    return {"held_out_mean": sum(held) / len(held), "per_fold": held, "picks": picks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="prebuilt index dir")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.6)
    ap.add_argument("--mu", type=float, default=250.0)
    ap.add_argument("--rrf-k", type=float, default=60.0)
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--out", default="runs/tune_fusion.json")
    args = ap.parse_args()

    index = InvertedIndex.load(args.index)
    bm25.build(index)
    boolean_vsm.build(index)
    lm_dirichlet.build(index)

    queries = load_queries()
    qrels = load_qrels()
    folds = make_folds([q for q, _ in queries if q in qrels])

    t0 = time.perf_counter()
    cache = {}
    for qid, text in queries:
        if qid not in qrels:
            continue
        cache[qid] = raw_signals(index, text, args.k1, args.b, args.mu)
    print(f"cached raw signals for {len(cache)} topics in {time.perf_counter() - t0:.1f}s")

    rows = []

    def add(label, fuse):
        r = evaluate(fuse, cache, qrels, folds, index)
        r["label"] = label
        rows.append(r)
        print(f"  {label:34s} ndcg={r['full_ndcg']:.4f} map={r['full_map']:.4f} "
              f"spread={r['spread']:.4f} folds=[" +
              " ".join(f"{m:.3f}" for m in r["fold_means"]) + "]")

    print("\n-- single signals --")
    add("bm25", lambda bm, cos, lm: bm)
    add("vsm", lambda bm, cos, lm: cos)
    add("lm", lambda bm, cos, lm: lm)

    print("\n-- min-max blends (bm25 + vsm), the shipped family --")
    for w in (1.0, 0.9, 0.8, 0.7, 0.6):
        add(f"minmax bm={w:.1f} vsm={1 - w:.1f}",
            lambda bm, cos, lm, w=w: w * minmax(bm) + (1 - w) * minmax(cos))

    print("\n-- min-max 3-way --")
    for wb in (0.7, 0.6, 0.5, 0.4):
        for wl in (0.1, 0.2, 0.3, 0.4):
            wv = round(1.0 - wb - wl, 3)
            if wv < 0:
                continue
            add(f"minmax bm={wb:.1f} vsm={wv:.1f} lm={wl:.1f}",
                lambda bm, cos, lm, wb=wb, wv=wv, wl=wl:
                    wb * minmax(bm) + wv * minmax(cos) + wl * minmax(lm))

    print("\n-- RRF --")
    K, D = args.rrf_k, args.depth
    add("rrf bm+vsm", lambda bm, cos, lm:
        rank_reciprocal(bm, K, D) + rank_reciprocal(cos, K, D))
    add("rrf bm+lm", lambda bm, cos, lm:
        rank_reciprocal(bm, K, D) + rank_reciprocal(lm, K, D))
    add("rrf bm+vsm+lm", lambda bm, cos, lm:
        rank_reciprocal(bm, K, D) + rank_reciprocal(cos, K, D) + rank_reciprocal(lm, K, D))
    for wb in (2.0, 3.0, 4.0):
        add(f"rrf {wb:.0f}*bm+vsm+lm", lambda bm, cos, lm, wb=wb:
            wb * rank_reciprocal(bm, K, D) + rank_reciprocal(cos, K, D)
            + rank_reciprocal(lm, K, D))
        add(f"rrf {wb:.0f}*bm+vsm", lambda bm, cos, lm, wb=wb:
            wb * rank_reciprocal(bm, K, D) + rank_reciprocal(cos, K, D))

    rows.sort(key=lambda r: -r["full_ndcg"])
    print("\n-- ranked by full-dev nDCG@10 --")
    for r in rows[:12]:
        print(f"  {r['full_ndcg']:.4f}  spread={r['spread']:.4f}  {r['label']}")

    cv = cv_select(rows, folds)
    print(f"\nfold-separated held-out mean: {cv['held_out_mean']:.4f}")
    print("  per-fold picks:")
    for f, (m, p) in enumerate(zip(cv["per_fold"], cv["picks"])):
        print(f"    fold {f}: {m:.4f}  <- {p}")

    os.makedirs("runs", exist_ok=True)
    payload = {
        "params": vars(args),
        "cv": cv,
        "rows": [{k: v for k, v in r.items() if k != "per_query_ndcg"} for r in rows],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
