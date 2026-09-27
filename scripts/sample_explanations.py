"""Print a random sample of kept explanations for hand review.

Topics the judge rejected most often are oversampled: if the judge caught several
errors there, it probably missed a few too.

    uv run python scripts/sample_explanations.py            # 30 examples
    uv run python scripts/sample_explanations.py -n 10 --seed 7 --uniform
    uv run python scripts/sample_explanations.py --topic "por와 para"
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"


def load(name):
    return [json.loads(l) for l in (OUT / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topic", help="only topics containing this text")
    ap.add_argument("--uniform", action="store_true", help="don't oversample high-rejection topics")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    kept = load("sft_explanations.jsonl")
    rejected = collections.Counter(r["topic"] for r in load("rejected_explanations.jsonl"))
    if args.topic:
        kept = [r for r in kept if args.topic in r["meta"]["topic"]]

    # weight 1 + (rejections in that topic): a 4/8-rejected topic is 5x as likely
    rng = random.Random(args.seed)
    weights = [1 if args.uniform else 1 + rejected[r["meta"]["topic"]] for r in kept]
    picked, pool = [], list(range(len(kept)))
    while pool and len(picked) < args.n:
        i = rng.choices(pool, weights=[weights[j] for j in pool])[0]
        pool.remove(i)
        picked.append(kept[i])

    for k, r in enumerate(picked, 1):
        m = r["meta"]
        print(f"\n{'=' * 70}\n[{k}/{len(picked)}] {m['topic']}  (topic rejections: {rejected[m['topic']]}/8)")
        print(f"\nQ: {r['messages'][1]['content']}\n\nA: {r['messages'][2]['content']}")


if __name__ == "__main__":
    main()
