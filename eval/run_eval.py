"""Run the eval set against one or more Ollama models and score the answers.

Standard library only, so you can read every line without looking up a package.

    python eval/run_eval.py --models qwen3.5:4b gemma4:e4b

How it works:
  1. Load eval/eval_set.jsonl (one test case per line).
  2. For each model x test case, call Ollama's local HTTP API (/api/chat).
  3. Score each answer with cheap automatic checks:
       - `check`: does the answer contain the expected Spanish form(s)?
       - `korean`: is the answer actually written in Korean (when it should be)?
  4. Write raw results (JSONL) + a Markdown report you read yourself.
     The `rubric` items can't be graded by string matching: read those answers
     and judge them. That's the "human eval" part.
"""

import argparse
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
HANGUL = re.compile(r"[가-힣]")
LATIN_CHAR = r"[A-Za-zÀ-ÿ]"
LATIN = re.compile(LATIN_CHAR)
ENGLISH_WORDS = re.compile(r"\b(the|is|are|and|you|this|that|with|verb|sentence)\b", re.IGNORECASE)


# ---------- talking to Ollama ----------

def chat(host, model, system, user, timeout=300):
    """One non-streaming chat call. Returns (answer_text, stats_dict)."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,  # thinking models: skip the hidden reasoning, we grade the final answer
        # temperature 0 = always pick the most likely token, so runs are repeatable
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 1024},
    }
    try:
        data = _post(f"{host}/api/chat", body, timeout)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        body.pop("think")  # some models reject the `think` field entirely
        data = _post(f"{host}/api/chat", body, timeout)

    text = THINK_BLOCK.sub("", data["message"]["content"]).strip()
    # Ollama reports durations in nanoseconds
    tok_s = data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 1e-9)
    return text, {"tokens": data.get("eval_count", 0), "tok_per_s": round(tok_s, 1)}


def _post(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------- scoring ----------

def normalize(s):
    # NFC so "é" typed two different ways compares equal; drop markdown emphasis; lowercase.
    s = unicodedata.normalize("NFC", s).replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", s).lower()


def contains_phrase(text, phrase):
    # Whole-word match, accent-sensitive: "dé" must not match "de", "era" must not match "eras".
    # Boundaries are Latin letters only, NOT \w: Korean endings attach directly to the
    # answer ("hable예요"), and \w would count those Hangul syllables as part of the word.
    return re.search(rf"(?<!{LATIN_CHAR}){re.escape(normalize(phrase))}(?!{LATIN_CHAR})", text) is not None


# For already-correct sentences (check values = [the learner's sentence]). Grades the *verdict*:
#   - the answer says it's right: an affirmation in its first line, or "틀린/고칠 곳(부분·데)이 없어요" anywhere,
#     and no hedge like "거의 다 맞았어요 / 고쳐 볼게요" in the first line;
#   - nothing is actually changed: every "고친 문장:" line is the learner's sentence, and no 'x' → 'y' with x != y.
# Filling the 4-part template with the unchanged sentence is fine; inventing a fix is not.
AFFIRM = ["맞아요", "맞습니다", "맞는 문장", "맞고", "맞게 썼", "완벽", "자연스러운 문장", "자연스러워요"]
NO_ERROR = re.compile(r"(틀린|고칠)\s*(곳|부분|데)[은이가]?\s*(하나도\s*|전혀\s*)?없")
DENY = ["거의", "고쳐", "고치면", "틀렸", "안 맞", "맞지 않", "어색"]
FIX_LINE = re.compile(r"고친 (?:문장|글)\s*:\s*(.+)")
ARROW = re.compile(r"'([^'\n]+)'\s*(?:→|\$?\\rightarrow\$?)\s*'([^'\n]+)'")


def sentence_key(s):
    return re.sub(r"[\"'“”«».,!?¿¡\s]", "", normalize(s))


def says_correct(answer, sentence):
    first = normalize(answer.strip().split("\n")[0])
    affirms = any(a in first for a in AFFIRM) or NO_ERROR.search(normalize(answer))
    if not affirms or any(d in first for d in DENY):
        return False
    plain = answer.replace("*", "")
    if any(sentence_key(f) != sentence_key(sentence) for f in FIX_LINE.findall(plain)):
        return False
    return all(a.strip() == b.strip() for a, b in ARROW.findall(plain))


def learner_sentence(item):
    """The one quoted sentence a `correction` item asks about, or None (paragraphs, other categories)."""
    quoted = re.findall(r'"([^"]+)"', item["prompt"])
    return quoted[0] if item["category"] == "correction" and len(quoted) == 1 else None


def score_check(answer, check, sentence=None):
    if check["type"] == "says_correct":
        return says_correct(answer, check["values"][0])
    if sentence and says_correct(answer, sentence):
        return False  # "틀린 곳이 없어요" can't pass because the fixed form shows up in the practice exercise
    text = normalize(answer)
    hits = [v for v in check["values"] if contains_phrase(text, v)]
    if check["type"] == "contains_any":
        return len(hits) > 0
    if check["type"] == "contains_all":
        return len(hits) == len(check["values"])
    raise ValueError(f"unknown check type {check['type']}")


def is_korean(answer):
    """Heuristic: enough Hangul relative to Latin letters, and not English prose.

    Answers legitimately contain Spanish examples, so we can't demand 100% Hangul.
    """
    hangul, latin = len(HANGUL.findall(answer)), len(LATIN.findall(answer))
    ratio = hangul / max(hangul + latin, 1)
    return ratio >= 0.25 and len(ENGLISH_WORDS.findall(answer)) <= 3


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+")
    p.add_argument("--rescore", type=Path, help="re-grade saved answers in a results/<timestamp> dir; no model calls")
    p.add_argument("--eval", default=ROOT / "eval" / "eval_set.jsonl", type=Path)
    p.add_argument("--system", default=ROOT / "prompts" / "tutor_system.txt", type=Path)
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--only", help="run only ids starting with this prefix, e.g. conj")
    p.add_argument("--out", type=Path, help="results directory (default: results/<timestamp>)")
    args = p.parse_args()

    system = args.system.read_text(encoding="utf-8")
    items = [json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        items = [i for i in items if i["id"].startswith(args.only)]
    if args.rescore:
        return rescore(args.rescore, {i["id"]: i for i in items})
    if not args.models:
        p.error("--models is required unless --rescore is given")

    out_dir = args.out or ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for model in args.models:
        print(f"\n=== {model} ({len(items)} items) ===")
        with open(out_dir / f"{model.replace(':', '_').replace('/', '_')}.jsonl", "w", encoding="utf-8") as f:
            for item in items:
                t0 = time.time()
                answer, stats = chat(args.host, model, system, item["prompt"])
                row = {
                    "model": model,
                    "id": item["id"],
                    "category": item["category"],
                    "check_pass": score_check(answer, item["check"], learner_sentence(item)),
                    "korean_pass": is_korean(answer) if item["require_korean"] else None,
                    "seconds": round(time.time() - t0, 1),
                    **stats,
                    "answer": answer,
                    "rubric": item.get("rubric"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_rows.append(row)
                mark = "PASS" if row["check_pass"] else "fail"
                print(f"  {item['id']:10} {mark}  {answer[:60].replace(chr(10), ' ')!r}")

    report = build_report(all_rows, args.models)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report.split("## Answers")[0])
    print(f"Full report with every answer: {out_dir / 'report.md'}")


def rescore(results_dir, items_by_id):
    """Grading is cheap, generation is slow: re-apply the current checks to saved answers."""
    rows, models = [], []
    for path in sorted(results_dir.glob("*.jsonl")):
        file_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in file_rows:
            item = items_by_id.get(row["id"])
            if item is None:
                continue
            row["check_pass"] = score_check(row["answer"], item["check"], learner_sentence(item))
            row["korean_pass"] = is_korean(row["answer"]) if item["require_korean"] else None
            row["rubric"] = item.get("rubric")
            rows.append(row)
            if row["model"] not in models:
                models.append(row["model"])
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in file_rows), encoding="utf-8")
    report = build_report(rows, models)
    (results_dir / "report.md").write_text(report, encoding="utf-8")
    print(report.split("## Answers")[0])


def build_report(rows, models):
    lines = ["# Eval report", "", "## Scores", "",
             "| model | category | check pass | korean pass | avg tok/s |", "|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["category"])].append(r)
        groups[(r["model"], "ALL")].append(r)
    for (model, cat), rs in groups.items():
        chk = sum(r["check_pass"] for r in rs)
        kr = [r["korean_pass"] for r in rs if r["korean_pass"] is not None]
        kr_s = f"{sum(kr)}/{len(kr)}" if kr else "-"
        speed = sum(r["tok_per_s"] for r in rs) / len(rs)
        name = f"**{cat}**" if cat == "ALL" else cat
        lines.append(f"| {model} | {name} | {chk}/{len(rs)} | {kr_s} | {speed:.0f} |")

    # Side-by-side answers, grouped by question, so you can compare models directly.
    lines += ["", "## Answers", ""]
    by_id = defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["model"]] = r
    for item_id, per_model in by_id.items():
        first = next(iter(per_model.values()))
        lines.append(f"### {item_id} ({first['category']})")
        if first["rubric"]:
            lines.append(f"> **Rubric:** {first['rubric']}")
        for model in models:
            r = per_model.get(model)
            if r:
                flags = f"check={'✅' if r['check_pass'] else '❌'}"
                if r["korean_pass"] is not None:
                    flags += f" korean={'✅' if r['korean_pass'] else '❌'}"
                lines += [f"**{model}** ({flags})", "", "```", r["answer"], "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
