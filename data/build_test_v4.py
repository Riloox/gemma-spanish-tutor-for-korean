"""Round 4: a new held-out test set, built and frozen before any round-4 training.

Why a new one: the 70-item eval/test_set.jsonl can't show a real win (a true +5 points still comes out around
p≈0.5), and round 4 was designed after reading grpo3's failing answers on it, so it can't grade round 4 either.

Same teacher pipeline as data/build_teacher_items.py (gemma4:26b writes items, blind judges check them), with:
  - its own files (data/out/test4_*.jsonl) and its own topics, so it shares no batches with the training items
  - no rule judge: a test item needs a correct sentence, a learner's version and the fix, not the rule text
  - distance from training: dropped if exact or near (trigram Jaccard) to any training sentence (every generator
    item and every raw teacher item) or to a dev/test sentence, and near-duplicates within the set are dropped
  - one item is EITHER a wrong item or a correct one, never both: twins would make answers correlated, and the
    significance tests (eval/compare.py) treat items as independent

Output: eval/test_set_v4.jsonl, the same schema as eval/test_set.jsonl:
  wrong  category "correction", check contains_any [the fixed span with a neighbouring word, absent from the learner's sentence]
  ok     category "correction_ok", check says_correct [the sentence]
Once it exists, data/common.eval_sentences() bans its sentences from all training data.

Stages, each resumable:
  generate  -> data/out/test4_raw.jsonl
  judge     -> data/out/test4_judged.jsonl
  (top-up)  with --stage all: more batches for any error type short of 2 x PER_SIDE good items, then judge again
  build     -> eval/test_set_v4.jsonl (+ data/out/test4_rejects.jsonl)

    uv run python -m data.build_test_v4 --stage all
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from data.build_corrections import GENERATORS
from data.build_explanations import append_jsonl, chat, grams, jaccard, load_jsonl
from data.build_teacher_items import BLIND_PROMPT, FOCUS, GENERATE_PROMPT, LEADING, PER_BATCH, ask_json, parse_json, \
    structure
from data.build_teacher_items import RAW as TRAIN_RAW
from data.common import OUT_DIR, ROOT, _norm, eval_sentences, write_jsonl

sys.path.insert(0, str(ROOT / "eval"))
from run_eval import contains_phrase, normalize, sentence_key  # noqa: E402

RAW = OUT_DIR / "test4_raw.jsonl"
JUDGED = OUT_DIR / "test4_judged.jsonl"
REJECTS = OUT_DIR / "test4_rejects.jsonl"
TEST = ROOT / "eval" / "test_set_v4.jsonl"

PER_SIDE = 13  # per error type: 13 wrong + 13 correct items -> 208 items
# Batches per type, from round 3's pass rates (past_tense kept 37% of raw items, the others 70-88%) with margin
# for the stricter distance filter: we need 26 good items per type.
BATCHES = {"personal_a": 8, "si_clause": 8, "ser_estar": 8, "gustar": 7, "subjunctive": 7, "past_tense": 14,
           "agreement": 7, "article_gender": 8}
NEAR = 0.6  # trigram Jaccard at or above this to a training/dev/test sentence, or to another test item -> dropped

# Disjoint from build_teacher_items.TOPICS, so the test sentences talk about other things.
TOPICS = ["sports and the gym", "cooking at home", "pets and animals", "music and concerts", "the news",
          "a job interview", "moving to a new apartment", "a birthday party", "the bank and money",
          "public transport in the city", "a museum visit", "movies and series", "a phone call with a relative",
          "the neighbourhood", "a camping trip"]

# Wrappers for questions the teacher wrote leadingly ("틀린 것 같은데…"). Not build_corrections.PROMPTS: the
# fine-tuned models saw those in every training example.
WRAPPERS = ["{s} 이 문장 괜찮은지 봐 주세요.", "제가 쓴 문장인데 확인해 주실래요? {s}", "{s}\n맞게 쓴 건가요?",
            "오늘 배운 걸로 문장 만들어 봤어요: {s}", "이거 스페인어로 맞아요? {s}"]


# ---------------------------------------------------------------- stage 1: generate

def stage_generate(extra=None):
    """extra: {type: more batches}, for top-ups after a judge pass came up short."""
    done = {(r["type"], r["batch"]) for r in load_jsonl(RAW)}
    have = Counter(t for t, _ in done)
    want = {t: max(n, have[t]) + (extra or {}).get(t, 0) for t, n in BATCHES.items()}
    todo = [(t, b) for t, n in want.items() for b in range(n) if (t, b) not in done]
    print(f"  {len(done)} batches on disk, {len(todo)} to go")
    for i, (etype, b) in enumerate(todo, 1):
        rng = random.Random(f"test4-{etype}-{b}")
        focus = FOCUS[etype][(b + 3) % len(FOCUS[etype])]  # offset: batch 0 isn't round 3's batch 0 focus
        prompt = GENERATE_PROMPT.format(n=PER_BATCH, etype=etype.replace("_", " "), focus=focus,
                                        topics=", ".join(rng.sample(TOPICS, 3)))
        data = chat([{"role": "user", "content": prompt}], temperature=0.9, num_predict=4000)
        items = parse_json(data["message"]["content"], "[")
        items = [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []
        append_jsonl(RAW, {"type": etype, "batch": b, "focus": focus, "items": items})
        print(f"  [gen {i}/{len(todo)}] {etype:14} batch {b:3}: {len(items)} items")


# ---------------------------------------------------------------- distance from training

def training_grams():
    """Trigram sets of every sentence a round-1..4 model could have trained on, plus the dev/test sentences."""
    sents = set(eval_sentences())
    for gen in GENERATORS.values():
        for it in gen():
            sents.update((_norm(it.correct), _norm(it.incorrect)))
    for row in load_jsonl(TRAIN_RAW):
        for x in row["items"]:
            if isinstance(x, dict) and x.get("correct"):
                sents.add(_norm(str(x["correct"])))
    print(f"  {len(sents)} training/dev/test sentences to keep away from")
    return sents, [grams(s) for s in sents]


def candidates():
    """Raw items with their template and a reject reason (or None): structure, then distance."""
    sents, sent_grams = training_grams()
    kept_grams, out = [], []
    for row in load_jsonl(RAW):
        for k, x in enumerate(row["items"]):
            item = {"id": f"t4-{row['type']}-{row['batch']}-{k}", "type": row["type"],
                    **{f: str(x.get(f, "")).strip() for f in ("correct", "right", "wrong", "ko", "question")}}
            item.update(rule="-", rule_short="-", ok_note="-")  # test items don't need them; placeholders that
            item["tmpl"], why = structure({**item, "rule": "규칙", "ok_note": "칭찬"})  # pass structure()'s Korean check
            item["incorrect"] = item["tmpl"].replace("{X}", item["wrong"])
            if not why:
                g = grams(_norm(item["correct"]))
                if _norm(item["correct"]) in sents or _norm(item["incorrect"]) in sents:
                    why = "is a training/dev/test sentence"
                elif any(jaccard(g, t) >= NEAR for t in sent_grams):
                    why = "near a training/dev/test sentence"
                elif any(jaccard(g, t) >= NEAR for t in kept_grams):
                    why = "near another test item"
                else:
                    kept_grams.append(g)
            out.append({**item, "structural": why})
    return out


# ---------------------------------------------------------------- stage 2: judge

def stage_judge():
    done = {r["id"] for r in load_jsonl(JUDGED)}
    todo = [c for c in candidates() if not c["structural"] and c["id"] not in done]
    print(f"  {len(done)} judged, {len(todo)} to go")
    for i, c in enumerate(todo, 1):
        on_correct = ask_json(BLIND_PROMPT.format(s=c["correct"]))
        on_wrong = ask_json(BLIND_PROMPT.format(s=c["incorrect"]))
        append_jsonl(JUDGED, {"id": c["id"], "on_correct": on_correct, "on_wrong": on_wrong})
        print(f"  [judge {i}/{len(todo)}] {c['id']:26} {verdict(c, on_correct, on_wrong) or 'ok'}")


def verdict(c, on_correct, on_wrong):
    if not isinstance(on_correct, dict) or not isinstance(on_wrong, dict):
        return "judge output unparseable"
    if on_correct.get("correct") is not True:
        return "judge: correct sentence has an error"
    if on_wrong.get("correct") is not False:
        return "judge: learner sentence is fine"
    if sentence_key(str(on_wrong.get("fixed", ""))) != sentence_key(c["correct"]):
        return "judge: fixed it differently"
    return None


def good_items():
    """(items passing structure, distance and both judges and having a fix span; the rejected ones with a reason)."""
    judged = {r["id"]: r for r in load_jsonl(JUDGED)}
    good, bad = [], []
    for c in candidates():
        why = c["structural"]
        if not why:
            j = judged.get(c["id"])
            why = "not judged yet" if j is None else verdict(c, j["on_correct"], j["on_wrong"])
        if not why and fix_span(c) is None:
            why = "no span that tells the fix apart"
        (bad if why else good).append({**c, "reject_reason": why} if why else c)
    return good, bad


TOP_UPS, TOP_UP_BATCHES = 3, 4


def stage_top_up():
    """Generate and judge more batches for any type with fewer than 2 * PER_SIDE good items, up to TOP_UPS times."""
    for _ in range(TOP_UPS):
        good = Counter(c["type"] for c in good_items()[0])
        short = {t: TOP_UP_BATCHES for t in BATCHES if good[t] < 2 * PER_SIDE}
        if not short:
            return
        print(f"  top-up: good items per type {dict(good)}; {TOP_UP_BATCHES} more batches for {list(short)}")
        stage_generate(short)
        stage_judge()


# ---------------------------------------------------------------- stage 3: build

def fix_span(c):
    """What a right answer must contain: the fixed span, widened by a neighbouring word until it no longer occurs
    in the learner's sentence ("el" alone is everywhere; "el agua" isn't in "la agua ...")."""
    pre, post = c["tmpl"].split("{X}")
    before = re.findall(r"[\wÀ-ÿ]+", pre)
    after = re.findall(r"[\wÀ-ÿ]+", post)
    bad = normalize(c["incorrect"])
    options = [c["right"]]
    if after:
        options.append(c["correct"][len(pre):len(pre) + len(c["right"]) + post.index(after[0]) + len(after[0])])
    if before:
        start = pre.rindex(before[-1])
        options.append(c["correct"][start:len(pre) + len(c["right"])])
        if after:
            options.append(c["correct"][start:len(pre) + len(c["right"]) + post.index(after[0]) + len(after[0])])
    for o in options:
        o = o.strip(" ,.;:¿?¡!")
        if len(o) >= 4 and not contains_phrase(bad, o):
            return o
    return None


def user_message(c, sentence, rng):
    """The teacher's own question with the sentence in double quotes (run_eval.learner_sentence looks for exactly
    one quoted sentence), or a neutral wrapper if the question is leading or quotes something else."""
    q = c["question"].replace('"{s}"', "{s}").replace("“{s}”", "{s}")
    if LEADING.search(q) or '"' in q or q.count("{s}") != 1:
        q = rng.choice(WRAPPERS)
    return q.replace("{s}", f'"{sentence}"')


def stage_build():
    good, bad = good_items()

    rng = random.Random(41)
    rows, short = [], {}
    for etype in BATCHES:
        pool = [c for c in good if c["type"] == etype]
        rng.shuffle(pool)
        wrong, ok = pool[:PER_SIDE], pool[PER_SIDE:2 * PER_SIDE]
        if len(ok) < PER_SIDE:
            short[etype] = len(pool)
        for c in wrong:
            rows.append({"category": "correction", "error_type": etype,
                         "prompt": user_message(c, c["incorrect"], rng),
                         "check": {"type": "contains_any", "values": [fix_span(c)]},
                         "require_korean": True, "source_id": c["id"], "fix": c["correct"]})
        for c in ok:
            rows.append({"category": "correction_ok", "error_type": etype,
                         "prompt": user_message(c, c["correct"], rng),
                         "check": {"type": "says_correct", "values": [c["correct"]]},
                         "rubric": "The sentence is already correct. Say so; don't invent an error or rewrite it as a 'fix'.",
                         "require_korean": True, "source_id": c["id"]})
    n = Counter()
    for r in rows:
        kind = "corr" if r["category"] == "correction" else "ok"
        n[kind] += 1
        r["id"] = f"t4-{kind}-{n[kind]:03d}"
    rows = [{"id": r.pop("id"), **r} for r in rows]

    write_jsonl(rows, TEST)
    write_jsonl(bad, REJECTS)
    print(f"  {n['corr']} wrong + {n['ok']} correct items")
    if short:
        print(f"  WARNING: fewer than {2 * PER_SIDE} good items for {short}; run more generate batches")
    print("  rejected:", dict(Counter(re.sub(r"\d+ times", "N times", b["reject_reason"]) for b in bad).most_common()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["generate", "judge", "build", "all"], default="all")
    args = p.parse_args()
    if TEST.exists() and args.stage in ("build", "all"):
        raise SystemExit(f"{TEST.relative_to(ROOT)} already exists and is frozen; delete it by hand to rebuild")
    if args.stage in ("generate", "all"):
        stage_generate()
    if args.stage in ("judge", "all"):
        stage_judge()
    if args.stage == "all":
        stage_top_up()
    if args.stage in ("build", "all"):
        stage_build()


if __name__ == "__main__":
    main()
