"""GRPO round 2: keep only the RL prompts the current model still gets wrong sometimes.

A GRPO group where every answer earns the same reward has zero advantage and teaches nothing. By the
end of round 1, 77% of groups were like that (frac_reward_zero_std), mostly prompts the model now
always gets right. So before round 2 we ask the round-1 model each prompt K times (temperature 1.0,
like training) through Ollama, grade the answers with the training verifiers, and drop the prompts
it aced every time.

Prompts it failed every time are kept: they're the errors we most want fixed (v2 and round 1 both
praise subtle mistakes), and as the policy moves some of their answers start scoring, which gives
signal. Prompts whose sentence is in the dev or test set are dropped too (data/common.py).

Resumable: every graded prompt is appended to rl_difficulty.jsonl, and a re-run skips those.

    uv run python -m data.filter_rl                          # gemma4-tutor-grpo, K=4 -> data/out/rl_prompts_hard.jsonl
    uv run python -m data.filter_rl --model gemma4-tutor-v2 --k 8
"""

import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data.common import OUT_DIR, eval_sentences, is_contaminated, write_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import contains_phrase, normalize  # noqa: E402

from train.rewards import grade_correction  # noqa: E402

PROMPTS = OUT_DIR / "rl_prompts.jsonl"
SCORES = OUT_DIR / "rl_difficulty.jsonl"
HARD = OUT_DIR / "rl_prompts_hard.jsonl"


def sample(host, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 1.0, "num_ctx": 4096, "num_predict": max_tokens}}
    req = urllib.request.Request(f"{host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]


def grade(row, answer):
    if row["task"] == "correction":
        return grade_correction(answer, row["sentence"], row["correct"], row["right"], row["ok"])
    return float(contains_phrase(normalize(answer), row["form"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-tutor-grpo")
    ap.add_argument("--k", type=int, default=4, help="answers sampled per prompt")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (Ollama queues what it can't run)")
    ap.add_argument("--max-tokens", type=int, default=512, help="same limit as training's --max-completion")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    banned = eval_sentences()
    keep = [i for i, r in enumerate(rows) if not (r["task"] == "correction" and
                                                   (is_contaminated(r["sentence"], banned) or is_contaminated(r["correct"], banned)))]
    print(f"{len(rows)} prompts, {len(rows) - len(keep)} dropped as dev/test sentences")

    done = {}
    if SCORES.exists():
        for line in SCORES.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d["model"] == args.model and len(d["rewards"]) == args.k:
                done[d["index"]] = d["rewards"]
    todo = [i for i in keep if i not in done]
    print(f"{len(done)} already graded, {len(todo)} to go ({args.k} answers each, model {args.model})")

    def work(i):
        return i, [grade(rows[i], sample(args.host, args.model, rows[i]["prompt"], args.max_tokens)) for _ in range(args.k)]

    with ThreadPoolExecutor(args.workers) as pool, open(SCORES, "a", encoding="utf-8") as f:
        for n, (i, rewards) in enumerate(pool.map(work, todo), 1):
            done[i] = rewards
            f.write(json.dumps({"index": i, "model": args.model, "rewards": rewards}) + "\n")
            f.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} graded", flush=True)

    # Report per kind of prompt, then keep everything that isn't a perfect score every time.
    stats = defaultdict(Counter)
    hard = []
    for i in keep:
        r, rewards = rows[i], done[i]
        kind = r["task"] if r["task"] == "conjugation" else f"{r['error_type']}{' (ok)' if r['ok'] else ''}"
        level = "aced" if min(rewards) == 1 else "failed" if max(rewards) == 0 else "mixed"
        stats[kind][level] += 1
        if level != "aced":
            hard.append(r)
    print(f"\n{'prompt kind':24} {'aced':>5} {'mixed':>6} {'failed':>7}")
    for kind in sorted(stats):
        c = stats[kind]
        print(f"{kind:24} {c['aced']:5} {c['mixed']:6} {c['failed']:7}")
    total = Counter()
    for c in stats.values():
        total.update(c)
    print(f"{'ALL':24} {total['aced']:5} {total['mixed']:6} {total['failed']:7}")
    write_jsonl(hard, HARD)


if __name__ == "__main__":
    main()
