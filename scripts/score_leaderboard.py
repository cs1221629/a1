"""
scripts/score_leaderboard.py -- project a harness report.json onto the
competition score, so experiments are compared on the number that is
actually graded rather than on nDCG alone.

    python scripts/score_leaderboard.py runs/final_report.json
    python scripts/score_leaderboard.py runs/a.json runs/b.json   # compare

THE FORMULA CHANGED between the dev leaderboard and the competition round.
It is no longer a weighted sum of the raw metrics. Re-fitted against all 46
rows of the Day-4 board (max residual 0.0136, mean 0.0029). Every row is
inside the +/-0.005 that 2-decimal rounding of the published Efficiency
column already explains, except Mohit Athikamsetty / Kathula Haasini, who
are an exact tie at nDCG 0.2168 to the published 4 d.p. -- so that pair is
display rounding in the nDCG column, not model error:

    score = 0.70 * percentile(nDCG@10) + 0.10 * percentile(MAP@10)
            + efficiency_modifier + index_size_score

    percentile(x) = (number of submissions strictly below x) / (n - 1)

This matches the assignment's own wording, "scaled primarily by percentile
rank among the whole class".

Why it matters far more than it sounds: the class's nDCG values are packed
into 0.1679-0.2307, so under the old raw formula the entire class was
separated by 0.044 of score. Ranking them turns that same tiny spread into
the full 0.70. One rank step is worth 0.70/45 = +0.0156, and around our
position +0.010 nDCG is worth about +0.218 of score -- more than twice the
whole efficiency range.

HONESTY NOTES
-------------
* The percentile is computed against a SNAPSHOT of the Day-4 field. Every
  other submission is a moving target, so a projection made today drifts as
  classmates resubmit. Treat direction as meaningful, absolute value as
  indicative.
* CLASS_MEDIAN_INDEX_BYTES is measured from the published index sizes.
* The build/query medians are ASSUMED, not derived: the leaderboard
  publishes only each row's rounded efficiency modifier, never the raw
  seconds, which is one equation in two unknowns.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.leaderboard import efficiency_modifier, index_size_score  # noqa: E402

NDCG_WEIGHT = 0.70
MAP_WEIGHT = 0.10

# Measured: median of the 46 index sizes published on the Day-4 board.
CLASS_MEDIAN_INDEX_BYTES = 15_610_754

# ASSUMED, not derived -- see the module docstring. Revised down from
# (50.0, 0.095): those were inferred when the board still reported raw dev
# metrics. Our Day-4 row pairs a ~33s hidden-corpus build and ~15ms latency
# with a published -0.02, which the reference modifier can only produce if
# the medians sit near these values.
ASSUMED_MEDIAN_BUILD_S = 15.0
ASSUMED_MEDIAN_QUERY_S = 0.020

# The dev corpus and the hidden corpus are different sizes, so a dev-run
# index size cannot be compared against the hidden class median directly.
# Calibrated from the two submissions measured on both. The current codec
# is the relevant one: it produces 17,004,159 B on dev and 12,847,008 B on
# the hidden corpus. (The pre-rewrite code gave 24,389,893 -> 17,755,184,
# a ratio of 0.728, so the mapping is stable to about 3%.)
DEV_TO_HIDDEN_INDEX_RATIO = 12_847_008 / 17_004_159

# Snapshot of the Day-4 competition field (nDCG@10, MAP@10) used only to
# turn our own metric into a percentile.
CLASS_NDCG = [
    0.2307, 0.2284, 0.2297, 0.2275, 0.2276, 0.2222, 0.2174, 0.2152, 0.2195,
    0.2147, 0.2123, 0.2188, 0.2168, 0.2223, 0.2149, 0.2122, 0.2168, 0.2088,
    0.2141, 0.2056, 0.2125, 0.2072, 0.2080, 0.2067, 0.2077, 0.2100, 0.2008,
    0.2051, 0.2081, 0.2007, 0.2034, 0.2079, 0.1960, 0.1953, 0.1952, 0.1944,
    0.1991, 0.1971, 0.2013, 0.1915, 0.1873, 0.1859, 0.1995, 0.1714, 0.1735,
    0.1679,
]
CLASS_MAP = [
    0.1286, 0.1253, 0.1254, 0.1254, 0.1276, 0.1253, 0.1229, 0.1201, 0.1233,
    0.1209, 0.1183, 0.1184, 0.1197, 0.1236, 0.1193, 0.1171, 0.1218, 0.1165,
    0.1171, 0.1180, 0.1177, 0.1164, 0.1174, 0.1156, 0.1166, 0.1172, 0.1150,
    0.1125, 0.1149, 0.1118, 0.1141, 0.1168, 0.1120, 0.1107, 0.1120, 0.1076,
    0.1105, 0.1108, 0.1147, 0.1068, 0.1067, 0.1014, 0.1009, 0.0937, 0.0931,
    0.0898,
]

# Our own Day-4 row, as the reference point every projection is measured from.
CURRENT = {
    "label": "submitted (Day-4 board, rank 23)",
    "ndcg": 0.2080, "map": 0.1174, "index_bytes": 12_847_008,
    "efficiency": -0.02,
}


def percentile(value, population):
    return sum(1 for x in population if x < value) / (len(population) - 1)


def project(ndcg, map10, index_bytes, efficiency):
    p_nd, p_mp = percentile(ndcg, CLASS_NDCG), percentile(map10, CLASS_MAP)
    size = index_size_score(index_bytes, CLASS_MEDIAN_INDEX_BYTES)
    return {
        "pct_ndcg": p_nd, "pct_map": p_mp,
        "ndcg_component": NDCG_WEIGHT * p_nd,
        "map_component": MAP_WEIGHT * p_mp,
        "efficiency": efficiency,
        "index_size_score": size,
        "score": NDCG_WEIGHT * p_nd + MAP_WEIGHT * p_mp + efficiency + size,
    }


def render(label, ndcg, map10, index_bytes, efficiency, eff_note):
    p = project(ndcg, map10, index_bytes, efficiency)
    print(f"\n{label}")
    print(f"  nDCG@10 {ndcg:.4f} -> percentile {p['pct_ndcg']:.3f} -> {p['ndcg_component']:+.4f}")
    print(f"  MAP@10  {map10:.4f} -> percentile {p['pct_map']:.3f} -> {p['map_component']:+.4f}")
    print(f"  efficiency {efficiency:+.4f}   {eff_note}")
    print(f"  index {index_bytes:,} B ({index_bytes / CLASS_MEDIAN_INDEX_BYTES:.3f}x median)"
          f" -> {p['index_size_score']:+.4f}")
    print(f"  PROJECTED SCORE = {p['score']:.4f}")
    return p


def main():
    ap = argparse.ArgumentParser(description="Project onto the competition score.")
    ap.add_argument("reports", nargs="*", help="harness report.json paths")
    ap.add_argument("--hidden-ndcg", type=float, default=None,
                    help="override the hidden-set nDCG instead of scaling the dev one")
    args = ap.parse_args()

    base = render(CURRENT["label"], CURRENT["ndcg"], CURRENT["map"],
                  CURRENT["index_bytes"], CURRENT["efficiency"], "(as published)")

    for path in args.reports:
        with open(path) as f:
            rep = json.load(f)
        agg, eff = rep["aggregate_metrics"], rep["efficiency"]
        # Dev metrics are on a different scale from the hidden set, so scale
        # the published hidden nDCG by the *relative* dev-set change rather
        # than pretending the dev number transfers directly.
        # The reference point must be the dev metrics of the code that
        # produced CURRENT's published hidden row, or the ratio silently
        # re-credits gains the board has already scored. That is the
        # post-rewrite submission: dev 0.680630 / 0.016940 -> hidden
        # 0.2080 / 0.1174. A run that reproduces those dev numbers
        # therefore projects to exactly the published hidden numbers,
        # which is the correct behaviour for a change that leaves the
        # ranking bit-identical and only moves efficiency.
        dev_ref_ndcg, dev_ref_map = 0.680630, 0.016940
        ndcg = args.hidden_ndcg or CURRENT["ndcg"] * (agg["ndcg@10"] / dev_ref_ndcg)
        map10 = CURRENT["map"] * (agg["map@10"] / dev_ref_map)
        hidden_bytes = int(eff["index_size_bytes"] * DEV_TO_HIDDEN_INDEX_RATIO)
        modifier = efficiency_modifier(eff["index_build_seconds"],
                                       eff["mean_query_latency_seconds"],
                                       ASSUMED_MEDIAN_BUILD_S,
                                       ASSUMED_MEDIAN_QUERY_S)
        p = render(f"{os.path.basename(path)}  (dev nDCG {agg['ndcg@10']:.4f} "
                   f"-> hidden est. {ndcg:.4f})",
                   ndcg, map10, hidden_bytes, modifier,
                   "[ESTIMATE: assumed class medians]")
        print(f"  delta vs. submitted: {p['score'] - base['score']:+.4f}")

    print("\nPercentiles are against a Day-4 SNAPSHOT of a moving field; the")
    print("efficiency modifier uses ASSUMED build/query medians. See docstring.")


if __name__ == "__main__":
    main()
