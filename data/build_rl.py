"""Step 5: prompts for RL (GRPO). No answers here, only questions plus what a verifier needs to grade one.

SFT shows the model one right answer per question. GRPO samples several answers to the same
question, grades each with a program, and pushes the model toward the better ones. So every
row carries its gold facts instead of a target text:

  correction   sentence the learner wrote, the correct sentence, the right/wrong forms, ok (no error?)
  conjugation  the expected form (anti-regression: SFT already got 24/24, RL shouldn't lose it)

Corrections are half wrong, half already correct, and each correct one is the twin of a wrong one
(same item, one word differs). A policy that always says "맞아요" or always "fixes" something
earns about half the reward, so only looking at the word pays.

Sentences come from the same generators as Step 2's correction data (correct by construction),
minus anything in the eval set.

    uv run python -m data.build_rl
"""

import json
import random
import re

from data.build_corrections import GENERATORS, PROMPTS
from data.common import OUT_DIR, SYSTEM_PROMPT, eval_sentences, is_contaminated, write_jsonl

SEED = 11
N_ITEMS_PER_TYPE = 150  # each item gives a wrong prompt and, with P_TWIN, its correct twin
P_TWIN = 1.0
CONJ_SHARE = 0.15  # of the final rows
BOLD = re.compile(r"\*\*([^*]+)\*\*")


def prompt(user):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def correction_rows(rng):
    banned = eval_sentences()
    rows = []
    for name, gen in GENERATORS.items():
        clean = [it for it in gen() if not (is_contaminated(it.incorrect, banned) or is_contaminated(it.correct, banned))]
        for it in rng.sample(clean, min(N_ITEMS_PER_TYPE, len(clean))):
            base = {"task": "correction", "error_type": it.type, "correct": it.correct, "right": it.right, "wrong": it.wrong}
            rows.append({"prompt": prompt(rng.choice(PROMPTS).format(s=it.incorrect)),
                         "sentence": it.incorrect, "ok": False, **base})
            if rng.random() < P_TWIN:
                rows.append({"prompt": prompt(rng.choice(PROMPTS).format(s=it.correct)),
                             "sentence": it.correct, "ok": True, **base})
        print(f"  {name:15} {len(clean):4} candidates")
    return rows


def conjugation_rows(rng, n):
    """Reuse Step 2's single-form questions; the gold form is in the SFT answer."""
    rows = []
    for line in (OUT_DIR / "sft_conjugation.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        kind, (_, user, assistant) = r["meta"]["kind"], r["messages"]
        if kind == "conj_short":
            form = assistant["content"].strip()
        elif kind == "conj_answer":
            form = BOLD.findall(assistant["content"])[0]
        else:
            continue
        rows.append({"prompt": prompt(user["content"]), "task": "conjugation", "form": form})
    return rng.sample(rows, min(n, len(rows)))


def main():
    rng = random.Random(SEED)
    corr = correction_rows(rng)
    conj = conjugation_rows(rng, round(len(corr) * CONJ_SHARE / (1 - CONJ_SHARE)))
    # Keep one schema across tasks: the dataset needs every column on every row.
    keys = ["task", "error_type", "sentence", "correct", "right", "wrong", "ok", "form"]
    rows = [{"prompt": r["prompt"], **{k: r.get(k) for k in keys}} for r in corr + conj]
    for r in rows:
        r["ok"] = bool(r["ok"])
        for k in keys:
            if r[k] is None:
                r[k] = ""
    rng.shuffle(rows)
    n_ok = sum(r["ok"] for r in rows)
    print(f"{len(corr)} correction prompts ({n_ok} already correct), {len(conj)} conjugation prompts")
    write_jsonl(rows, OUT_DIR / "rl_prompts.jsonl")


if __name__ == "__main__":
    main()
