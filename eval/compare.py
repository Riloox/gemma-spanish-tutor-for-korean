"""Is model B really better than model A, or is it noise? Paired comparison of two eval runs.

Re-running the eval can't measure noise: it uses temperature 0, so every rerun gives the same
answers. The noise that matters is *which items* happened to be in the set. Both models answered
the same items, so we only look at the items where they disagree:

  b = A passed, B failed        c = A failed, B passed

If B were no better, each disagreement would be a coin flip. McNemar's exact test asks how
surprising c-vs-b is under that coin flip. 86 vs 83 could be c=5, b=2, and p≈0.45: not evidence.
The bootstrap interval resamples the items to show how far the score difference could move.

Standard library only, like run_eval.py.

    python eval/compare.py results/A/gemma4-tutor-grpo.jsonl results/B/gemma4-tutor-grpo2.jsonl \
        --eval eval/test_set.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def mcnemar_p(b, c):
    """Two-sided exact p-value: the chance of a split at least this lopsided from fair coin flips."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(pairs, reps=10_000, seed=0):
    """95% interval for B's pass rate minus A's, resampling items with replacement."""
    rng = random.Random(seed)
    n = len(pairs)
    diffs = sorted(sum(bp - ap for ap, bp in (pairs[rng.randrange(n)] for _ in range(n))) / n for _ in range(reps))
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def compare(a_path, b_path, eval_path):
    a, b = load(a_path), load(b_path)
    items = {}
    if eval_path:
        items = {json.loads(line)["id"]: json.loads(line)
                 for line in Path(eval_path).read_text(encoding="utf-8").splitlines() if line.strip()}
    ids = [i for i in a if i in b]
    name_a, name_b = a[ids[0]]["model"], b[ids[0]]["model"]

    groups = defaultdict(list)
    for i in ids:
        pair = (int(a[i]["check_pass"]), int(b[i]["check_pass"]))
        groups["ALL"].append(pair)
        groups[f"category: {a[i]['category']}"].append(pair)
        if items.get(i, {}).get("error_type"):
            groups[f"error: {items[i]['error_type']}"].append(pair)

    lo, hi = bootstrap_ci(groups["ALL"])
    lines = [f"## {name_b} vs {name_a} ({len(ids)} items)", "",
             f"B minus A pass rate, 95% bootstrap interval: {lo:+.1%} to {hi:+.1%}"
             + ("  (includes 0: not a clear difference)" if lo <= 0 <= hi else ""), "",
             f"| group | {name_a} | {name_b} | only A passed | only B passed | McNemar p |",
             "|---|---|---|---|---|---|"]
    for key in ["ALL"] + sorted(k for k in groups if k != "ALL"):
        pairs = groups[key]
        only_a = sum(ap and not bp for ap, bp in pairs)
        only_b = sum(bp and not ap for ap, bp in pairs)
        n = len(pairs)
        lines.append(f"| {key} | {sum(ap for ap, _ in pairs)}/{n} | {sum(bp for _, bp in pairs)}/{n} "
                     f"| {only_a} | {only_b} | {mcnemar_p(only_a, only_b):.3f} |")
    flips = [f"- {i}: {'A' if a[i]['check_pass'] else 'B'} only" for i in ids if a[i]["check_pass"] != b[i]["check_pass"]]
    lines += ["", "Items where they disagree:", ""] + (flips or ["- none"]) + [""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", help="results jsonl of the baseline model")
    p.add_argument("b", help="results jsonl of the new model")
    p.add_argument("--eval", help="the eval file both ran on (adds a per-error-type breakdown if it has one)")
    args = p.parse_args()
    print(compare(args.a, args.b, args.eval))


if __name__ == "__main__":
    main()
