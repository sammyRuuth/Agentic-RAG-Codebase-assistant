"""Retrieval evaluation: dense-only vs BM25-only vs hybrid (RRF).

    uv run python -m eval.evaluate            # summary table
    uv run python -m eval.evaluate -v         # + per-query breakdown

Metrics (all at cutoff k, averaged over queries - "macro" average):
  Precision@k  = |relevant ∩ top-k| / k            "how much of what we returned is useful"
  Recall@k     = |relevant ∩ top-k| / |relevant|   "how much of what's useful did we find"
  Hit@k        = 1 if any relevant chunk in top-k  "did we find anything at all"
  MRR          = mean of 1 / rank of first relevant chunk  "how high is the first good one"

Note that Precision@10 has a ceiling: a query with 3 relevant chunks can score
at most 0.3. With ~4 labels per query the best possible P@10 here is ~0.4,
so P@10 must be read together with Recall@10.
"""

import argparse
import json
from pathlib import Path

from retrieval.hybrid import HybridRetriever

QUERIES = Path(__file__).with_name("queries.json")
MODES = ("dense", "bm25", "hybrid")


def score(retrieved: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = retrieved[:k]
    hits = [i for i, r in enumerate(top, start=1) if r in relevant]
    return {
        "P": len(hits) / k,
        "R": len(hits) / len(relevant),
        "Hit": 1.0 if hits else 0.0,
        "MRR": 1.0 / hits[0] if hits else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    data = json.loads(QUERIES.read_text())
    r = HybridRetriever()
    missing = {i for q in data for i in q["relevant"]} - r.docs.keys()
    assert not missing, f"labels reference unknown chunk ids (re-index?): {missing}"

    totals = {m: {"P": 0.0, "R": 0.0, "Hit": 0.0, "MRR": 0.0} for m in MODES}
    for q in data:
        rel = set(q["relevant"])
        row = []
        for mode in MODES:
            s = score(r.search_ids(q["query"], k=args.k, mode=mode), rel, args.k)
            for key in s:
                totals[mode][key] += s[key]
            row.append(f"{mode} R={s['R']:.2f}")
        if args.verbose:
            print(f"{q['query'][:70]:<70}  |rel|={len(rel)}  " + "  ".join(row))

    n = len(data)
    ceiling = sum(min(len(q["relevant"]), args.k) for q in data) / (n * args.k)
    print(f"\n{n} queries, k={args.k}  (max achievable P@{args.k} given labels: {ceiling:.2f})\n")
    print(f"{'mode':<8} {'P@'+str(args.k):>7} {'R@'+str(args.k):>7} {'Hit@'+str(args.k):>7} {'MRR':>7}")
    for mode in MODES:
        t = {key: v / n for key, v in totals[mode].items()}
        print(f"{mode:<8} {t['P']:>7.3f} {t['R']:>7.3f} {t['Hit']:>7.3f} {t['MRR']:>7.3f}")


if __name__ == "__main__":
    main()
