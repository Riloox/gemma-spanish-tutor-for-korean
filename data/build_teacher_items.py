"""Step 6a: new correction items written by the local teacher (gemma4:26b), checked before use.

Why: the generators in build_corrections.py are correct by construction but narrow. personal_a has 7 frames,
all "Veo/Visito/Llamo … at the start of the sentence"; si_clause has 1. After two GRPO rounds the model still
praises personal-a errors in questions or after "voy a", and si-clauses it hasn't seen. It learned the frames,
not the rule. In round 1, 41-44% of those prompts scored 0 on all 4 samples: no signal for RL to use.

The teacher writes *items*, not answers: a correct sentence, the span that a learner gets wrong, the Korean
translation, the rule, and a learner question to wrap it in. Answers are then rendered by the same check-first
templates as the generator items (build_corrections.render_wrong / render_ok), so the format is identical.

Nothing the teacher writes is trusted until it passes:
  structure   `right` occurs once in `correct`; swapping in `wrong` gives the learner's sentence; Korean fields are Korean
  decontam    not the same as, nor close to (trigram Jaccard), any dev or test sentence
  blind judge two fresh calls, one sentence each, no hint which is which: the correct one must be judged correct,
              and the wrong one judged wrong *with the same fix* (an independent route to the same answer)
  rule judge  the rule, short rule, praise line and translation are checked for false claims

Stages, each resumable (re-running skips what's on disk):
  generate  -> data/out/teacher_raw.jsonl
  judge     -> data/out/teacher_judged.jsonl
  filter    -> data/out/teacher_items.jsonl (+ teacher_rejects.jsonl with a reason each)
  render    -> data/out/sft_teacher_corrections.jsonl

    uv run python -m data.build_teacher_items --stage all
    uv run python -m data.build_teacher_items --stage all --scale 0.05     # smoke test: ~5% of the batches
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

from data.build_corrections import GENERATORS, Item, render_ok, render_wrong
from data.build_explanations import append_jsonl, chat, grams, jaccard, load_jsonl
from data.common import OUT_DIR, _norm, eval_sentences, is_contaminated, write_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import sentence_key  # noqa: E402

RAW = OUT_DIR / "teacher_raw.jsonl"
JUDGED = OUT_DIR / "teacher_judged.jsonl"
ITEMS = OUT_DIR / "teacher_items.jsonl"
REJECTS = OUT_DIR / "teacher_rejects.jsonl"
SFT_OUT = OUT_DIR / "sft_teacher_corrections.jsonl"

PER_BATCH = 8
OVERSAMPLE = 1.8  # raw items requested per item wanted; the filter is strict
TARGET = {  # items wanted per type, weighted to where round 1 had no RL signal
    "personal_a": 260, "si_clause": 260, "ser_estar": 240, "gustar": 200,
    "subjunctive": 160, "past_tense": 140, "agreement": 60, "article_gender": 60,
}
TWIN_SHARE = 0.35  # of items also rendered as the correct sentence ("맞아요"), next to their broken twin
NEAR_DUP = 0.6     # trigram Jaccard to a dev/test sentence at or above this -> dropped

# What to vary, per type. Described in words, not example sentences, so the teacher can't echo the eval.
FOCUS = {
    "personal_a": [
        "a person as direct object inside a yes/no question (tú or usted)",
        "a person as direct object after a periphrasis: ir a / querer / tener que / poder + infinitive",
        "a person as direct object in the preterite or imperfect, with various subjects (nosotros, ellos, ella)",
        "indefinite people as direct object: todos, alguien, nadie, cada uno, a stranger",
        "a + el must contract to al before a masculine person (al profesor, al médico)",
        "verbs that take people as objects: conocer, ayudar, cuidar, escuchar, extrañar, abrazar, entender, saludar, "
        "acompañar, esperar",
        "the other direction: a thing, place or animal-as-thing object that must NOT take a (the learner added one)",
        "tener and hay never take personal a with people (the learner added one), e.g. number of siblings or friends",
    ],
    "si_clause": [
        "real condition: si + present indicative, result in present, future or imperative; learner used si + future",
        "real condition: learner used si + present subjunctive after si",
        "unreal present: si + imperfect subjunctive, result in conditional; learner used conditional after si",
        "unreal present: learner put the result clause in future or present instead of conditional",
        "unreal past: si + hubiera + participle, result habría + participle; learner used habría after si",
        "the si-clause comes second in the sentence (result first), any of the above patterns",
        "questions and negatives with si-clauses, varied persons (usted, ustedes, nosotros)",
    ],
    "ser_estar": [
        "professions and roles take ser, with varied subjects incl. plural; learner used estar",
        "location of an event (party, exam, wedding, meeting) takes ser; learner used estar",
        "location of a person or thing takes estar; learner used ser",
        "temporary states and feelings (cansado, nervioso, enfermo, ocupado, aburrido) take estar; learner used ser",
        "estar + gerund for actions in progress; learner used ser",
        "inherent characteristics, nationality, material, possession take ser; learner used estar",
        "time, dates and days take ser (son las…, hoy es…); learner used estar",
        "results and current conditions (la puerta está abierta, la tienda está cerrada); learner used ser",
    ],
    "gustar": [
        "plural subject after gustar-type verbs (gustan, encantan, interesan, molestan, duelen, faltan, quedan); "
        "learner used the singular",
        "gustar in the preterite or imperfect with a plural subject (gustaron, gustaban); learner used singular",
        "the pronoun doesn't match the a-phrase: the learner wrote e.g. me or les where a singular third person needs le, "
        "or le where a nosotros needs nos (the a-phrase itself is right)",
        "learner made the person the subject (yo gusto / tú encantas…) instead of me/te gusta",
        "infinitive or singular noun subject takes singular gusta; learner used plural",
        "questions with gustar-type verbs (¿Te gustan…? ¿Les interesa…?)",
    ],
    "subjunctive": [
        "cuando / en cuanto / hasta que + future-time event takes subjunctive; learner used indicative or future",
        "para que / antes de que always take subjunctive; learner used indicative",
        "no creer que / dudar que take subjunctive; learner used indicative",
        "emotion triggers (me alegra que, siento que, me molesta que) take subjunctive",
        "the other direction: creer que / pensar que / es verdad que take indicative; learner used subjunctive",
        "ojalá and es posible que with varied persons",
    ],
    "past_tense": [
        "a completed event with a time marker takes the preterite; learner used the imperfect",
        "background description, age, time of day in the past take the imperfect; learner used the preterite",
        "an action in progress interrupted by another (mientras / cuando); one verb in the wrong tense",
        "repeated past habits (siempre, todos los días, de niño) take the imperfect; learner used the preterite",
    ],
    "agreement": [
        "adjective number and gender after a plural or feminine noun, nouns with irregular gender (la mano, el día, "
        "el mapa, la foto)",
        "predicate adjective after ser/estar agreeing with a plural or feminine subject",
        "demonstratives and possessives agreeing with the noun (estas, nuestras, aquellos)",
    ],
    "article_gender": [
        "nouns whose gender doesn't follow -o/-a (el problema, la mano, el día, la flor, el planeta, la gente)",
        "feminine nouns starting with stressed a- take el in the singular (el agua, el águila) but stay feminine",
        "indefinite articles and contractions (un/una, al, del) with irregular-gender nouns",
    ],
}

TOPICS = ["work and the office", "family", "travel and transport", "food and restaurants", "studying Spanish",
          "friends and weekend plans", "health and the doctor", "shopping", "living abroad in a Spanish-speaking country",
          "Korean culture explained to a Spanish speaker", "school and university", "weather and seasons", "hobbies"]

GENERATE_PROMPT = """You write practice data for a Spanish tutor app whose users are Korean adults learning Spanish.
Write {n} items of the error type "{etype}". Focus for this batch: {focus}.
Everyday topics; use these where they fit: {topics}.

Each item is one natural, fully correct Spanish sentence plus the single mistake a Korean learner would make in it.
Rules:
- `correct`: standard Latin American Spanish (ustedes, not vosotros), 5-16 words, natural and fully correct.
- `right`: the exact span of `correct` that the learner gets wrong. It must appear exactly once in `correct`.
- `wrong`: what the learner wrote instead. Replacing `right` by `wrong` once must give the learner's sentence.
  Keep spans minimal but whole words (e.g. "a mi tía" vs "mi tía", "están" vs "son").
- The learner's sentence must be clearly ungrammatical in standard Spanish, not merely a different meaning,
  a regional variant or a style choice. Avoid adjectives whose meaning changes with ser/estar (listo, malo, rico...).
- Every item uses a different sentence frame: vary subjects, persons, tenses, questions vs statements, word order.
- `ko`: Korean translation of `correct`, 해요체 (never 합니다/습니다).
- `rule`: 2-3 sentences in Korean 해요체 that explain why `right` and not `wrong`, naming the actual words.
  Explain the rule, not just the answer. No English. Where it helps, say how this differs from Korean grammar.
- `rule_short`: one short Korean line (under 40 characters) naming the rule and the right form, e.g. "사람 목적어 앞 → a".
  It is shown before the verdict, so it must not say whether the learner was right.
- `ok_note`: one Korean 해요체 sentence praising what a learner who wrote `correct` did right, specific to this rule.
- `question`: a message a Korean learner might send with the sentence, containing the placeholder {{s}} exactly once.
  Vary the tone (polite, casual community-board style, short, with some context about why they are asking).
  It must not hint at the error.

Format example (a different error type; `right` is in `correct`, `wrong` is what the learner wrote):
{{"correct": "Mis tías son muy simpáticas.", "right": "simpáticas", "wrong": "simpáticos", "ko": "제 이모들은 아주 친절해요.",
 "rule": "형용사는 꾸미거나 설명하는 명사의 성과 수에 맞춰요. tías는 여성 복수라서 simpáticas를 써요. 한국어 형용사는 모양이 바뀌지 않아서 놓치기 쉬워요.",
 "rule_short": "tías(여성 복수) → simpáticas", "ok_note": "여성 복수 주어에 맞춰 형용사를 simpáticas로 잘 썼어요.",
 "question": "이모들 소개하는 문장 써 봤어요. {{s}} 괜찮아요?"}}

Output only a JSON array of {n} objects with keys: correct, right, wrong, ko, rule, rule_short, ok_note, question."""

BLIND_PROMPT = """You are a strict Spanish teacher. Is the following sentence grammatically correct standard Spanish?
Ignore capitalization and punctuation style. If it has an error, correct it changing as little as possible.
Output JSON only: {{"correct": true or false, "fixed": "the corrected sentence, or the same sentence if correct"}}

Sentence: {s}"""

RULE_PROMPT = """You are a strict Spanish grammar checker. A tutor wrote the Korean notes below about the sentence
"{correct}", which learners often get wrong as "{incorrect}". Check every claim about Spanish grammar in them.
Report only real errors: a false claim about Spanish, or a translation that changes the meaning. Ignore style and nuance.
Output JSON only: {{"errors": ["short description", ...]}} (an empty list if there are none)

Translation of the correct sentence: {ko}
Why it's "{right}" and not "{wrong}": {rule}
Short rule: {rule_short}
Praise for a learner who wrote the correct sentence: {ok_note}"""

HANGUL = re.compile(r"[가-힣]")
LATIN_WORD = r"[A-Za-zÀ-ÿ]"


def parse_json(text, opener):
    closer = {"[": "]", "{": "}"}[opener]
    m = re.search(re.escape(opener) + r".*" + re.escape(closer), text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- stage 1: generate

def batches(scale):
    out = []
    for etype, want in TARGET.items():
        n = max(1, math.ceil(want * OVERSAMPLE / PER_BATCH * scale))
        out += [(etype, b, FOCUS[etype][b % len(FOCUS[etype])]) for b in range(n)]
    return out


def stage_generate(scale):
    done = {(r["type"], r["batch"]) for r in load_jsonl(RAW)}
    todo = [b for b in batches(scale) if (b[0], b[1]) not in done]
    print(f"  {len(done)} batches on disk, {len(todo)} to go")
    for i, (etype, b, focus) in enumerate(todo, 1):
        rng = random.Random(f"{etype}-{b}")
        prompt = GENERATE_PROMPT.format(n=PER_BATCH, etype=etype.replace("_", " "), focus=focus,
                                        topics=", ".join(rng.sample(TOPICS, 3)))
        data = chat([{"role": "user", "content": prompt}], temperature=0.9, num_predict=4000)
        items = parse_json(data["message"]["content"], "[")
        items = [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []
        append_jsonl(RAW, {"type": etype, "batch": b, "focus": focus, "items": items})
        print(f"  [gen {i}/{len(todo)}] {etype:14} batch {b:3}: {len(items)} items")


# ---------------------------------------------------------------- structure + decontamination

def candidates():
    """Flatten raw batches into items: id, template ({X} where the error goes), learner's sentence, and a structural
    reject reason (or None)."""
    banned = eval_sentences()
    banned_grams = [grams(s) for s in banned]
    seen, out = set(), []
    for row in load_jsonl(RAW):
        for k, x in enumerate(row["items"]):
            item = {"id": f"{row['type']}-{row['batch']}-{k}", "type": row["type"], "focus": row["focus"],
                    **{f: str(x.get(f, "")).strip() for f in
                       ("correct", "right", "wrong", "ko", "rule", "rule_short", "ok_note", "question")}}
            item["tmpl"], why = structure(item)
            item["incorrect"] = item["tmpl"].replace("{X}", item["wrong"])
            if not why:
                key = sentence_key(item["correct"])
                if key in seen:
                    why = "duplicate"
                elif is_contaminated(item["correct"], banned) or is_contaminated(item["incorrect"], banned):
                    why = "in dev/test set"
                elif any(jaccard(grams(_norm(item["correct"])), g) >= NEAR_DUP for g in banned_grams):
                    why = "near a dev/test sentence"
                seen.add(key)
            out.append({**item, "structural": why})
    return out


def structure(x):
    """(template with {X} at the error span, reject reason or None)."""
    c, r, w = x["correct"], x["right"], x["wrong"]
    if not (c and r and w) or r == w:
        return "", "missing or identical spans"
    # whole-word occurrences only: "a" must not match inside "casa"
    hits = [m.start() for m in re.finditer(rf"(?<!{LATIN_WORD}){re.escape(r)}(?!{LATIN_WORD})", c)]
    if len(hits) != 1:
        return "", f"`right` occurs {len(hits)} times in `correct`"
    tmpl = c[:hits[0]] + "{X}" + c[hits[0] + len(r):]
    if sentence_key(tmpl.replace("{X}", w)) == sentence_key(c):
        return "", "learner sentence equals the correct one"
    if not 4 <= len(c.split()) <= 18:
        return tmpl, "length"
    if x["question"].count("{s}") != 1:
        return tmpl, "question lacks a single {s}"
    for f in ("ko", "rule", "ok_note", "question"):  # rule_short may be all Spanish: "a ti → te"
        if not HANGUL.search(x[f]):
            return tmpl, f"{f} not Korean"
    if re.search(r"\b(the|is|are|and|this|that)\b", x["rule"], re.IGNORECASE):
        return tmpl, "English in rule"
    return tmpl, None


# ---------------------------------------------------------------- stage 2: judge

def ask_json(prompt, opener="{"):
    data = chat([{"role": "user", "content": prompt}], temperature=0, num_predict=500)
    return parse_json(data["message"]["content"], opener)


def stage_judge():
    done = {r["id"] for r in load_jsonl(JUDGED)}
    todo = [c for c in candidates() if not c["structural"] and c["id"] not in done]
    print(f"  {len(done)} judged, {len(todo)} to go")
    for i, c in enumerate(todo, 1):
        on_correct = ask_json(BLIND_PROMPT.format(s=c["correct"]))
        on_wrong = ask_json(BLIND_PROMPT.format(s=c["incorrect"]))
        rule = ask_json(RULE_PROMPT.format(**c))
        append_jsonl(JUDGED, {"id": c["id"], "on_correct": on_correct, "on_wrong": on_wrong, "rule": rule})
        print(f"  [judge {i}/{len(todo)}] {c['id']:22} {verdict(c, on_correct, on_wrong, rule) or 'ok'}")


def verdict(c, on_correct, on_wrong, rule):
    """None if the item passes every judge, else why not."""
    if not isinstance(on_correct, dict) or not isinstance(on_wrong, dict) or not isinstance(rule, dict):
        return "judge output unparseable"
    if on_correct.get("correct") is not True:
        return "judge: correct sentence has an error"
    if on_wrong.get("correct") is not False:
        return "judge: learner sentence is fine"
    if sentence_key(str(on_wrong.get("fixed", ""))) != sentence_key(c["correct"]):
        return "judge: fixed it differently"
    if rule.get("errors"):
        return "judge: rule text has errors"
    return None


# ---------------------------------------------------------------- stage 3: filter

def stage_filter():
    judged = {r["id"]: r for r in load_jsonl(JUDGED)}
    good, bad = [], []
    for c in candidates():
        why = c["structural"]
        if not why:
            j = judged.get(c["id"])
            why = "not judged yet" if j is None else verdict(c, j["on_correct"], j["on_wrong"], j["rule"])
            if why == "judge: rule text has errors":
                c = {**c, "rule_errors": j["rule"]["errors"]}
        (bad if why else good).append({**c, "reject_reason": why} if why else c)
    write_jsonl(good, ITEMS)
    write_jsonl(bad, REJECTS)
    kept = Counter(g["type"] for g in good)
    total = Counter(c["type"] for c in good + bad)
    for t in TARGET:
        print(f"  {t:15} kept {kept[t]:4} / {total[t]:4} raw   (target {TARGET[t]})")
    print("  rejected:", dict(Counter(re.sub(r"\d+ times", "N times", b["reject_reason"]) for b in bad).most_common()))


# ---------------------------------------------------------------- stage 4: render

# A question that presupposes the verdict ("틀린 것 같아요", "왜 이렇게 써야 하나요?") would teach the model to answer the
# question instead of reading the sentence, and twins share their question. Those fall back to the stock wrappers.
LEADING = re.compile(r"틀린 것 같|틀렸|왜|고쳐야|부분|자리|써야 (하나요|해요|돼요)|맞는 이유")


def question(x, sentence):
    q = x["question"]
    return None if LEADING.search(q) else q.replace("{s}", sentence)


def to_item(x):
    return Item(x["type"], x["tmpl"], x["right"], x["wrong"], x["ko"], x["rule"], x["rule_short"], x["ok_note"])


def stage_render():
    rng = random.Random(23)
    rows = load_jsonl(ITEMS)
    pools = {t: [to_item(x) for x in rows if x["type"] == t] + gen() for t, gen in GENERATORS.items()}
    out = []
    for x in rows:
        item, pool = to_item(x), pools[x["type"]]
        assert item.incorrect == x["incorrect"], x["id"]
        out.append(render_wrong(rng, item, pool, user=question(x, item.incorrect)))
        if rng.random() < TWIN_SHARE:
            out.append(render_ok(rng, item, user=question(x, item.correct)))
    for r in out:
        r["meta"]["source"] = "teacher"
    rng.shuffle(out)
    write_jsonl(out, SFT_OUT)
    print("  kinds:", dict(Counter(r["meta"]["kind"] for r in out)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["generate", "judge", "filter", "render", "all"], default="all")
    p.add_argument("--scale", type=float, default=1.0, help="fraction of the generation batches (smoke tests)")
    args = p.parse_args()
    if args.stage in ("generate", "all"):
        stage_generate(args.scale)
    if args.stage in ("judge", "all"):
        stage_judge()
    if args.stage in ("filter", "all"):
        stage_filter()
    if args.stage in ("render", "all"):
        stage_render()


if __name__ == "__main__":
    main()
