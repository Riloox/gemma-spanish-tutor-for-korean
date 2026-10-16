"""Step 5: the verifiers GRPO learns from. Plain functions: an answer in, a number out.

They reuse the eval's graders (eval/run_eval.py), so "a correct verdict" means the same thing in
training and in the eval. The eval set itself is never used here (data/build_rl.py bans its sentences).

correctness  (the one that matters, 0..1)
  already-correct sentence  1 if the answer says it's right and changes nothing (the eval's says_correct)
  sentence with an error    1 if a "고친 문장:" line is exactly the correct sentence
                            0.5 if it doesn't claim "맞아요" and the right form is in a fix line or 'x' → 'y'
                            0 if it never gives the fix; 0 or -praise_penalty if it praises the sentence (v2's failure)
  conjugation               1 if the expected form appears as a whole word
check        (round 3, 0..1) the "쓴 형태: '<form>' ✓/✗" line quotes the learner's form with the right mark
rewrite      (round 4, replaces correctness for corrections, 0..1) 0.7 if the "다시 써 보면:" line is exactly the
             correct sentence (the learner's own sentence when it has no error), + 0.3 if the verdict agrees
             (says it's right on a correct sentence). Praising a wrong sentence is 0, whatever the rewrite. One number for both failures: inventing a fix and missing one.
hygiene      (penalties, 0 or negative)
  -1 for running into the length limit (v1 looped until 1,024 tokens), -0.5 for not answering in Korean
  (checked on corrections only; "답만" conjugation answers are a bare Spanish word)

A policy that always says "맞아요" scores ~0.5 on corrections (half the prompts are twins of wrong
ones); one that always "fixes" scores at most ~0.5. Only reading the word gets to 1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import (ARROW, FIX_LINE, USED_LINE, contains_phrase, is_korean, normalize,  # noqa: E402
                      affirms, rewrites, says_correct, sentence_key)

REWRITE_MATCH, VERDICT = 0.7, 0.3


def grade_correction(answer, sentence, correct, right, ok, praise_penalty=0.0):
    if ok:
        return 1.0 if says_correct(answer, sentence) else 0.0
    if says_correct(answer, sentence):
        return -praise_penalty
    plain = answer.replace("*", "")
    fixes = FIX_LINE.findall(plain)
    if any(sentence_key(f) == sentence_key(correct) for f in fixes):
        return 1.0
    # Partial credit only where the fix is stated (fix line or 'x' → 'y'), not for the form showing up in the exercise
    stated = " ".join(fixes + [b for _, b in ARROW.findall(plain)])
    return 0.5 if contains_phrase(normalize(stated), right) else 0.0


def grade_rewrite(answer, sentence, correct, ok):
    """Round 4: rewrite first, then compare. `correct` is the gold sentence (== `sentence` when ok).
    Praising a wrong sentence scores 0 even under a right rewrite: "네, 맞아요!" on an error is the failure that kept
    grpo2 and grpo3 level with the base, and a learner reads the verdict first."""
    if not ok and affirms(answer):
        return 0.0
    found = rewrites(answer.replace("*", ""))
    match = bool(found) and sentence_key(found[0]) == sentence_key(correct)
    verdict = says_correct(answer, sentence) if ok else True
    return REWRITE_MATCH * match + VERDICT * verdict


def grade_check(answer, wrong, right, ok):
    """Check-first answers open with "쓴 형태: '<form>' ✓/✗". 1 if that line quotes what the learner wrote and marks
    it right; 0 if missing or wrong. The verdict is graded separately; this pays for looking before judging."""
    m = USED_LINE.search(answer.replace("*", ""))
    if not m:
        return 0.0
    used, mark = normalize(m.group(1)), ("✓" if ok else "✗")
    return float(mark in used and ("✗" if ok else "✓") not in used and contains_phrase(used, right if ok else wrong))


def make_rewards(tokenizer, max_completion_length, praise_penalty=0.0, check=False, rewrite=False):
    """The reward functions GRPOTrainer calls, bound to the tokenizer: correctness, hygiene and, with check=True,
    check (grade_check; round 3, after SFT v3 taught the check-first format). rewrite=True grades corrections
    with grade_rewrite instead (round 4, after SFT v4 taught the rewrite-first format).

    They decode completion_ids themselves: TRL's Gemma 4 response parser returns empty content when the
    answer ends in "<turn|>
<eos>", and every answer would grade as blank.
    """
    def texts(completion_ids):
        return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    def correctness(completion_ids, task, sentence, correct, right, ok, form, **_):
        if rewrite:
            return [grade_rewrite(a, s, cor, o) if t == "correction" else float(contains_phrase(normalize(a), f))
                    for a, t, s, cor, o, f in zip(texts(completion_ids), task, sentence, correct, ok, form)]
        return [grade_correction(a, s, cor, r, o, praise_penalty) if t == "correction"
                else float(contains_phrase(normalize(a), f))
                for a, t, s, cor, r, o, f in zip(texts(completion_ids), task, sentence, correct, right, ok, form)]

    def hygiene(completion_ids, task, **_):
        out = []
        for a, ids, t in zip(texts(completion_ids), completion_ids, task):
            p = -1.0 if len(ids) >= max_completion_length else 0.0
            if t == "correction" and not is_korean(a):
                p -= 0.5
            out.append(p)
        return out

    def check_first(completion_ids, task, wrong, right, ok, **_):
        return [grade_check(a, w, r, o) if t == "correction" else 0.0
                for a, t, w, r, o in zip(texts(completion_ids), task, wrong, right, ok)]

    return [correctness, hygiene] + ([check_first] if check else [])
