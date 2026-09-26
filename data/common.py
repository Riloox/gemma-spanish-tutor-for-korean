"""Shared helpers for the data builders."""

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "out"
SYSTEM_PROMPT = (ROOT / "prompts" / "tutor_system.txt").read_text(encoding="utf-8").strip()


# ---------- Korean grammar helpers ----------
# Korean particles depend on whether the previous syllable ends in a consonant (받침):
#   책 + 은/이/을  (has 받침)     vs     커피 + 는/가/를  (no 받침)

def has_batchim(word):
    last = word.rstrip("*'\" .,!?)")[-1]
    if "가" <= last <= "힣":
        return (ord(last) - ord("가")) % 28 != 0
    # A Spanish word read in Korean: only final n / l / m become a 받침 (hable→하블레, pan→판, sol→솔).
    return last.lower() in "nlm"


def josa(word, pair):
    """josa("책", "은/는") -> "책은"; josa("커피", "을/를") -> "커피를"; josa("pan", "으로/로") -> "pan으로"."""
    with_b, without_b = pair.split("/")
    if pair == "으로/로" and _ends_in_rieul(word):
        return word + "로"  # ㄹ 받침 takes 로, not 으로: 서울로, sol로
    return word + (with_b if has_batchim(word) else without_b)


def _ends_in_rieul(word):
    last = word.rstrip("*'\" .,!?)")[-1]
    if "가" <= last <= "힣":
        return (ord(last) - ord("가")) % 28 == 8
    return last.lower() == "l"


def copula(word):
    """'**hable**' + 예요, '**pan**' + 이에요 — the polite 'it is' ending."""
    return "이에요" if has_batchim(word.strip("*'\"")) else "예요"


# ---------- output ----------

def example(user, assistant, **meta):
    """One SFT training example in the standard chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": meta,
    }


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples -> {path.relative_to(ROOT)}")


# ---------- decontamination: never train on the exam ----------

def _norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"[^\w\s]", "", s).strip()


def eval_sentences():
    """Every quoted Spanish sentence in the eval set, normalized."""
    found = set()
    for line in (ROOT / "eval" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            for q in re.findall(r'"([^"]+)"', json.loads(line)["prompt"]):
                found.add(_norm(q))
    return found


def is_contaminated(sentence, eval_set):
    return _norm(sentence) in eval_set
