"""Step 5: the verifiers GRPO learns from. Plain functions: an answer in, a number out.

They reuse the eval's graders (eval/run_eval.py), so "a correct verdict" means the same thing in
training and in the eval. The eval set itself is never used here (data/build_rl.py bans its sentences).

correctness  (the one that matters, 0..1)
  already-correct sentence  1 if the answer says it's right and changes nothing (the eval's says_correct)
  sentence with an error    1 if a "고친 문장:" line is exactly the correct sentence
                            0.5 if it doesn't claim "맞아요" and the right form shows up somewhere
                            0 if it praises the sentence (v2's failure) or never gives the fix
  conjugation               1 if the expected form appears as a whole word
hygiene      (penalties, 0 or negative)
  -1 for running into the length limit (v1 looped until 1,024 tokens), -0.5 for not answering in Korean
  (checked on corrections only; "답만" conjugation answers are a bare Spanish word)

A policy that always says "맞아요" scores ~0.5 on corrections (half the prompts are twins of wrong
ones); one that always "fixes" scores at most ~0.5. Only reading the word gets to 1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import FIX_LINE, contains_phrase, is_korean, normalize, says_correct, sentence_key  # noqa: E402


def grade_correction(answer, sentence, correct, right, ok):
    if ok:
        return 1.0 if says_correct(answer, sentence) else 0.0
    if says_correct(answer, sentence):
        return 0.0
    fixes = FIX_LINE.findall(answer.replace("*", ""))
    if any(sentence_key(f) == sentence_key(correct) for f in fixes):
        return 1.0
    return 0.5 if contains_phrase(normalize(answer), right) else 0.0


def make_rewards(tokenizer, max_completion_length):
    """The two reward functions GRPOTrainer calls, bound to the tokenizer.

    They decode completion_ids themselves: TRL's Gemma 4 response parser returns empty content when the
    answer ends in "<turn|>
<eos>", and every answer would grade as blank.
    """
    def texts(completion_ids):
        return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    def correctness(completion_ids, task, sentence, correct, right, ok, form, **_):
        return [grade_correction(a, s, cor, r, o) if t == "correction" else float(contains_phrase(normalize(a), f))
                for a, t, s, cor, r, o, f in zip(texts(completion_ids), task, sentence, correct, right, ok, form)]

    def hygiene(completion_ids, task, **_):
        out = []
        for a, ids, t in zip(texts(completion_ids), completion_ids, task):
            p = -1.0 if len(ids) >= max_completion_length else 0.0
            if t == "correction" and not is_korean(a):
                p -= 0.5
            out.append(p)
        return out

    return [correctness, hygiene]
