"""Step 2c: teacher-written explanations, distilled from a bigger local model.

Distillation = a big model (teacher) writes answers, the small model (student) learns
to imitate them. The teacher runs locally in Ollama, so it's free and private.

Four stages, each resumable (re-running skips work already on disk):
  questions  teacher turns each seed topic into realistic learner questions
  answers    teacher answers each question with the same tutor system prompt
  review     teacher re-reads each answer as a strict grammar checker (LLM-as-a-judge)
  filter     automatic quality gates + judge verdict -> data/out/sft_explanations.jsonl
             (+ rejected_explanations.jsonl with the reason for each rejection)

    uv run python -m data.build_explanations --stage all
"""

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request

from data.common import OUT_DIR, SYSTEM_PROMPT, example, write_jsonl
from data.topics import PERSONAS, TOPICS

TEACHER = "gemma4:26b"
HOST = "http://localhost:11434"
QUESTIONS_PER_TOPIC = 8
RAW_Q = OUT_DIR / "raw_questions.jsonl"
RAW_A = OUT_DIR / "raw_answers.jsonl"
RAW_R = OUT_DIR / "raw_reviews.jsonl"
EVAL_FILE = OUT_DIR.parent.parent / "eval" / "eval_set.jsonl"

QUESTION_PROMPT = """한국인 스페인어 학습자가 실제로 물어볼 법한 질문을 {n}개 만들어 주세요.

주제: {topic}
질문자 예시: {personas}

조건:
- 질문자마다 수준과 말투를 다르게 해 주세요 (정중한 존댓말, 커뮤니티 게시판 말투, 짧은 질문, 사정을 길게 설명하는 질문).
- 일부 질문에는 학습자가 직접 쓴 스페인어 단어나 예문을 넣어 주세요.
- 질문만 쓰고, 답은 쓰지 마세요.
- 출력은 JSON 문자열 배열 하나만: ["질문1", "질문2", ...]"""


# Written in English on purpose: instructions for the *judge* aren't training data, and the
# model follows precise checking instructions more reliably in English. Tested in step 2c:
# it caught "Ella está listo" (-> lista) and "Espero que vienes" (-> vengas), and passed correct text.
JUDGE_PROMPT = """You are a strict Spanish grammar checker. Below is an answer a Spanish tutor wrote for a Korean learner.
Check EVERY Spanish word, form and sentence in it, including examples and exercises: gender/number agreement
(including agreement with the subject inside exercise sentences), verb conjugations, accents, ser/estar, and any
factual claim about Spanish grammar or pronunciation. Ignore style and the Korean text itself. A blank (___) in an
exercise is not an error, but the words around it must be correct.
Output JSON only: {"errors": ["short description", ...]}  (empty list if there are none)

ANSWER:
"""


# ---------------------------------------------------------------- Ollama client

def chat(messages, temperature, num_predict, retries=3):
    body = {
        "model": TEACHER, "messages": messages, "stream": False, "think": False, "keep_alive": "30m",
        "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": num_predict},
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{HOST}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    request failed ({e}); retry {attempt + 1}/{retries} in 30s")
            time.sleep(30)
    raise RuntimeError("Ollama kept failing; is it running?")


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- similarity (dedup + decontamination)

def grams(s):
    s = re.sub(r"\s+", "", s.lower())
    return {s[i:i + 3] for i in range(len(s) - 2)}


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def eval_prompt_grams():
    return [grams(json.loads(l)["prompt"]) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- stage 1: questions

def stage_questions():
    done = {r["topic_id"] for r in load_jsonl(RAW_Q)}
    rng = random.Random(11)
    for tid, (pain, topic) in enumerate(TOPICS):
        if tid in done:
            continue
        personas = ", ".join(rng.sample(PERSONAS, 4))
        prompt = QUESTION_PROMPT.format(n=QUESTIONS_PER_TOPIC, topic=topic, personas=personas)
        t0 = time.time()
        data = chat([{"role": "user", "content": prompt}], temperature=0.9, num_predict=1500)
        text = data["message"]["content"]
        match = re.search(r"\[.*\]", text, re.DOTALL)
        try:
            questions = [q.strip() for q in json.loads(match.group(0)) if isinstance(q, str) and q.strip()] if match else []
        except json.JSONDecodeError:
            questions = []
        append_jsonl(RAW_Q, {"topic_id": tid, "pain_point": pain, "topic": topic, "questions": questions})
        print(f"  [q {tid + 1}/{len(TOPICS)}] {len(questions)} questions, {time.time() - t0:.0f}s  {topic[:40]}")


def unique_questions():
    """Flatten, drop near-duplicates and anything too close to an eval question."""
    eval_g = eval_prompt_grams()
    kept, kept_g, dropped_eval = [], [], 0
    for row in load_jsonl(RAW_Q):
        for q in row["questions"]:
            g = grams(q)
            if any(jaccard(g, e) > 0.45 for e in eval_g):
                dropped_eval += 1
                continue
            if any(jaccard(g, k) > 0.8 for k in kept_g):
                continue
            kept.append({"qid": f"{row['topic_id']}-{len(kept)}", "pain_point": row["pain_point"], "topic": row["topic"], "question": q})
            kept_g.append(g)
    return kept, dropped_eval


# ---------------------------------------------------------------- stage 2: answers

def stage_answers(limit=None):
    questions, dropped_eval = unique_questions()
    done = {r["qid"] for r in load_jsonl(RAW_A)}
    todo = [q for q in questions if q["qid"] not in done][:limit]
    print(f"  {len(questions)} unique questions ({dropped_eval} dropped as too close to eval), {len(done)} answered, {len(todo)} to go")
    for i, q in enumerate(todo, 1):
        t0 = time.time()
        data = chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q["question"]}],
                    temperature=0.7, num_predict=1500)
        append_jsonl(RAW_A, {**q, "answer": data["message"]["content"].strip(), "done_reason": data.get("done_reason"),
                             "tokens": data.get("eval_count"), "seconds": round(time.time() - t0, 1)})
        print(f"  [a {i}/{len(todo)}] {data.get('eval_count')} tok, {time.time() - t0:.0f}s  {q['question'][:50]!r}")


# ---------------------------------------------------------------- stage 3: review (LLM-as-a-judge)

def stage_review():
    done = {r["qid"] for r in load_jsonl(RAW_R)}
    todo = [a for a in load_jsonl(RAW_A) if a["qid"] not in done and a.get("done_reason") == "stop"]
    print(f"  {len(done)} reviewed, {len(todo)} to go")
    for i, a in enumerate(todo, 1):
        t0 = time.time()
        data = chat([{"role": "user", "content": JUDGE_PROMPT + clean(a["answer"])}], temperature=0, num_predict=600)
        text = data["message"]["content"]
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            errors = json.loads(match.group(0))["errors"] if match else None
        except (json.JSONDecodeError, KeyError, TypeError):
            errors = None  # unparseable verdict -> treated as a rejection in the filter
        append_jsonl(RAW_R, {"qid": a["qid"], "errors": errors, "raw": text if errors is None else None})
        verdict = "unparseable" if errors is None else (f"{len(errors)} error(s)" if errors else "clean")
        print(f"  [r {i}/{len(todo)}] {verdict}, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- stage 4: filter

LATEX = [(re.compile(r"\$\\(rightarrow|to|Rightarrow)\$"), "→"), (re.compile(r"\$\\(leftarrow|Leftarrow)\$"), "←")]


def clean(text):
    """Fix harmless formatting artifacts before judging/keeping (the teacher sometimes writes LaTeX arrows)."""
    for pattern, repl in LATEX:
        text = pattern.sub(repl, text)
    return text.strip()


HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
ENGLISH = re.compile(r"\b(the|is|are|and|you|this|that|with|verb|sentence)\b", re.IGNORECASE)
RED_FLAGS = [  # claims the spec forbids, or assistant artifacts
    (re.compile(r"1\s*:\s*1로?\s*(똑같|같|대응)"), "claims 1:1 vowel mapping"),
    (re.compile(r"완전히 (똑같|같)아요"), "claims sounds are identical"),
    (re.compile(r"AI|인공지능|언어 모델"), "talks about being an AI"),
    (re.compile(r"<think>|</think>"), "thinking leaked"),
    (re.compile(r"\$\\[a-zA-Z]+"), "leftover LaTeX"),
]


def reject_reason(row):
    a = row["answer"]
    if row.get("done_reason") != "stop":
        return "truncated (hit token limit)"
    if not 150 <= len(a) <= 4000:
        return f"length {len(a)}"
    h, l = len(HANGUL.findall(a)), len(LATIN.findall(a))
    if h / max(h + l, 1) < 0.35:
        return "not mainly Korean"
    if len(ENGLISH.findall(a)) > 3:
        return "English prose"
    for pattern, why in RED_FLAGS:
        for m in pattern.finditer(a):
            # "1:1로 똑같지 않아요" is the *correct* teaching; only flag un-negated claims.
            if not re.search(r"않|아니|없", a[m.end():m.end() + 8]):
                return why
    return None


def stage_filter():
    rows = [{**r, "answer": clean(r["answer"])} for r in load_jsonl(RAW_A)]
    reviews = {r["qid"]: r["errors"] for r in load_jsonl(RAW_R)}
    good, bad = [], []
    for r in rows:
        why = reject_reason(r)
        if not why:
            errors = reviews.get(r["qid"], "missing")
            if errors == "missing":
                why = "not reviewed yet"
            elif errors is None:
                why = "judge verdict unparseable"
            elif errors:
                why = "judge found errors"
                r = {**r, "judge_errors": errors}
        if why:
            bad.append({**r, "reject_reason": why})
        else:
            good.append(example(r["question"], r["answer"], kind="explanation", pain_point=r["pain_point"], topic=r["topic"]))
    write_jsonl(good, OUT_DIR / "sft_explanations.jsonl")
    write_jsonl(bad, OUT_DIR / "rejected_explanations.jsonl")
    reasons = {}
    for b in bad:
        reasons[b["reject_reason"]] = reasons.get(b["reject_reason"], 0) + 1
    print(f"  kept {len(good)}/{len(rows)}; rejected: {reasons}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["questions", "answers", "review", "filter", "all"], default="all")
    p.add_argument("--limit", type=int, help="answer at most N questions (for smoke tests)")
    args = p.parse_args()
    if args.stage in ("questions", "all"):
        stage_questions()
    if args.stage in ("answers", "all"):
        stage_answers(args.limit)
    if args.stage in ("review", "all"):
        stage_review()
    if args.stage in ("filter", "all"):
        stage_filter()


if __name__ == "__main__":
    main()
