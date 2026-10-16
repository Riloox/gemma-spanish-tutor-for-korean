"""Round 4: correction SFT data written by the stock model itself, in the rewrite-first format, kept only if right.

Why: every SFT round so far cost recall. v3 took the stock model from 36/40 to 27/40 on the test set's corrections:
template answers pull a 2B model toward text it would never write, and it loses some of what it knew. Here the
stock gemma4:e2b writes the answers. It gets the rewrite-first format (build_corrections.rewrite_lines) through
a format note and two worked examples, answers each prompt up to K times, and a sample is kept only if
train/rewards.grade_rewrite gives it full marks, its "바뀐 곳" line agrees, it is in Korean and short enough.
The kept answers are stored under the plain tutor system prompt (no note, no examples), so SFT teaches the
format and the verdicts the model can already reach, in its own words.

Prompts the stock model never gets right in K tries are the ones that matter most, so they still go in, rendered
from the item by the template (build_corrections.render_wrong_v4 / render_ok_v4, correct by construction).

Items: the generators plus the checked teacher items, N_PER_TYPE per error type, each asked as the learner's
wrong sentence AND as its correct twin (50/50, and the model has to read the word that differs). Plus N_MULTI
paragraph items ("고친 글", already rewrite-first) from the templates. Nothing in the dev or test sets.

Resumable: every sample is appended to data/out/selfdistill_samples.jsonl.

    uv run python -m data.build_selfdistill                  # -> data/out/sft_corrections_v4.jsonl
"""

import argparse
import json
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data.build_corrections import GENERATORS, PROMPTS, render_multi, render_ok_v4, render_wrong_v4
from data.build_explanations import load_jsonl
from data.build_teacher_items import ITEMS as TEACHER_ITEMS
from data.build_teacher_items import question, to_item
from data.common import OUT_DIR, SYSTEM_PROMPT, eval_sentences, example, is_contaminated, write_jsonl
from data.filter_rl import sample

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import ARROW, is_korean  # noqa: E402

from train.rewards import grade_rewrite  # noqa: E402

SAMPLES = OUT_DIR / "selfdistill_samples.jsonl"
OUT = OUT_DIR / "sft_corrections_v4.jsonl"
SEED = 31
N_PER_TYPE = 150
N_MULTI = 260
MAX_TOKENS = 320  # longer answers are dropped: GRPO generates up to 384, and a cut-off answer is penalized

FORMAT_NOTE = """

학습자가 쓴 스페인어 문장을 확인할 때는 항상 이 두 줄로 시작합니다:
**다시 써 보면:** (학습자의 문장 전체를 정확한 스페인어로 다시 씁니다. 틀린 곳만 최소한으로 고치고, 틀린 곳이 없으면 한 글자도 바꾸지 않고 그대로 씁니다.)
**바뀐 곳:** 'before' → 'after' (바꾼 곳이 없으면 '없음')
그다음, 바뀐 곳이 없으면 '네, 맞아요!'로 시작해 번역과 잘한 점을 짧게 씁니다. 바뀐 곳이 있으면 고친 문장, 왜 그런지(규칙), 짧은 연습 문제 하나를 씁니다. 전체 답은 짧게 씁니다."""


def items_by_type():
    banned = eval_sentences()
    raw = load_jsonl(TEACHER_ITEMS)
    teacher, raw_teacher = [to_item(x) for x in raw], {x["correct"]: x for x in raw}
    pools = {}
    for name, gen in GENERATORS.items():
        pool = [it for it in gen() + [t for t in teacher if t.type == name]
                if not (is_contaminated(it.incorrect, banned) or is_contaminated(it.correct, banned))]
        pools[name] = pool
    return pools, raw_teacher


def build_prompts(rng):
    """Deterministic list of prompts (the resume file indexes into it) and two few-shot examples kept out of it."""
    pools, raw_teacher = items_by_type()
    shots = [render_wrong_v4(rng, pools["agreement"][0], pools["agreement"]),
             render_ok_v4(rng, pools["gustar"][0])]
    shot_sentences = {pools["agreement"][0].correct, pools["gustar"][0].correct}
    prompts = []
    for name, pool in pools.items():
        chosen = rng.sample([it for it in pool if it.correct not in shot_sentences], min(N_PER_TYPE, len(pool) - 1))
        for it in chosen:
            x = raw_teacher.get(it.correct)
            for sentence, ok in ((it.incorrect, False), (it.correct, True)):
                user = (question(x, sentence) if x else None) or rng.choice(PROMPTS).format(s=sentence)
                prompts.append({"item": it, "pool": pool, "user": user, "sentence": sentence, "ok": ok})
    print(f"  {len(prompts)} prompts: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()) + " items in pools")
    return prompts, shots, pools


def messages(shots, user):
    turns = [{"role": "system", "content": SYSTEM_PROMPT + FORMAT_NOTE}]
    for s in shots:
        turns += s["messages"][1:]
    return turns + [{"role": "user", "content": user}]


def accept(p, data):
    """Why this sample can't be kept, or None."""
    answer = data["message"]["content"]
    if data.get("done_reason") == "length" or data.get("eval_count", 0) > MAX_TOKENS:
        return "too long"
    if grade_rewrite(answer, p["sentence"], p["item"].correct, p["ok"]) < 0.999:
        return "wrong rewrite or verdict"
    changed = [ln for ln in answer.replace("*", "").splitlines() if ln.strip().startswith("바뀐 곳")]
    if not changed:
        return "no 바뀐 곳 line"
    if p["ok"] != ("없음" in changed[0] and not ARROW.search(changed[0])):
        return "바뀐 곳 disagrees"
    if not is_korean(answer):
        return "not Korean"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e2b")
    ap.add_argument("--k", type=int, default=4, help="tries per prompt; stops at the first kept one")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    args = ap.parse_args()

    rng = random.Random(SEED)
    prompts, shots, pools = build_prompts(rng)

    tries = {}
    for r in load_jsonl(SAMPLES):
        if r["model"] == args.model and r["index"] < len(prompts) and r["sentence"] == prompts[r["index"]]["sentence"]:
            tries.setdefault(r["index"], []).append(r)
    todo = [i for i in range(len(prompts))
            if not any(t["reject"] is None for t in tries.get(i, [])) and len(tries.get(i, [])) < args.k]
    print(f"  {len(prompts) - len(todo)} prompts settled, {len(todo)} to go (up to {args.k} tries each, {args.model})")

    lock = threading.Lock()
    f = open(SAMPLES, "a", encoding="utf-8")

    def work(i):
        p = prompts[i]
        for _ in range(args.k - len(tries.get(i, []))):
            data = sample(args.host, args.model, messages(shots, p["user"]), MAX_TOKENS + 80, args.temperature,
                          full=True)
            row = {"index": i, "model": args.model, "sentence": p["sentence"], "answer": data["message"]["content"],
                   "tokens": data.get("eval_count", 0), "reject": accept(p, data)}
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                tries.setdefault(i, []).append(row)
            if row["reject"] is None:
                return

    with ThreadPoolExecutor(args.workers) as pool:
        for n, _ in enumerate(pool.map(work, todo), 1):
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} prompts sampled", flush=True)
    f.close()

    rows, stats, reasons = [], Counter(), Counter()
    for i, p in enumerate(prompts):
        it, kind = p["item"], f"{p['item'].type}{' (ok)' if p['ok'] else ''}"
        kept = next((t for t in tries.get(i, []) if t["reject"] is None), None)
        reasons.update(t["reject"] for t in tries.get(i, []) if t["reject"])
        if kept:
            rows.append(example(p["user"], kept["answer"].strip(), kind="corr_ok" if p["ok"] else "corr_wrong",
                                error_type=it.type, format="rewrite", source="self"))
            stats[(kind, "self")] += 1
        else:
            rows.append(render_ok_v4(rng, it, user=p["user"]) if p["ok"]
                        else render_wrong_v4(rng, it, p["pool"], user=p["user"]))
            rows[-1]["meta"]["source"] = "template"
            stats[(kind, "template")] += 1

    everything = [it for pool in pools.values() for it in pool]
    for _ in range(N_MULTI):
        parts = rng.sample(everything, rng.choice([2, 3]))
        flags = [rng.random() < 0.7 for _ in parts]
        if not any(flags):
            flags[0] = True
        rows.append(render_multi(rng, list(zip(parts, flags))))

    print(f"\n  {'prompt kind':24} {'self':>5} {'template':>9}")
    for kind in sorted({k for k, _ in stats}):
        print(f"  {kind:24} {stats[(kind, 'self')]:5} {stats[(kind, 'template')]:9}")
    print(f"  rejected samples: {dict(reasons.most_common())}")
    rng.shuffle(rows)
    write_jsonl(rows, OUT)


if __name__ == "__main__":
    main()
