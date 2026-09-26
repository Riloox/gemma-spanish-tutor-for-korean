"""Build conjugation SFT examples. Every verb form comes from data/conjugator.py.

Four example types, each targeting something the baseline got wrong:
  short   - "X의 T, P 형태는? 답만"          -> just the form (eval-style drill)
  answer  - natural question                -> form + one-line context
  table   - "X T 변화표"                     -> full table (learners ask for this constantly)
  verify  - "X의 T, P 형태가 'Y' 맞나요?"    -> yes, or NO + which tense Y really is.
            Wrong candidates come from the *confusable* tense (indicative vs subjunctive,
            conditional vs imperfect...), the exact mix-ups we saw in Step 1.

    uv run python -m data.build_conjugation
"""

import json
import random

from data.common import OUT_DIR, copula, example, josa, write_jsonl
from data.conjugator import PERSON_KO, TENSES, UnknownVerb, conjugate

SEED = 42
N_SHORT, N_ANSWER, N_TABLE, N_VERIFY = 900, 600, 300, 700

VERBS = {
    "ser": "~이다", "estar": "(상태·위치가) ~이다/있다", "tener": "가지다", "hacer": "하다/만들다",
    "ir": "가다", "venir": "오다", "decir": "말하다", "poder": "~할 수 있다", "querer": "원하다",
    "saber": "(사실을) 알다", "conocer": "(사람·장소를) 알다", "dar": "주다", "ver": "보다",
    "poner": "놓다", "salir": "나가다", "llegar": "도착하다", "llevar": "가지고 가다",
    "dejar": "두다/남기다", "seguir": "계속하다/따라가다", "encontrar": "찾다/발견하다",
    "llamar": "부르다/전화하다", "pensar": "생각하다", "volver": "돌아오다", "tomar": "잡다/마시다",
    "hablar": "말하다", "comer": "먹다", "vivir": "살다", "beber": "마시다", "escribir": "쓰다",
    "leer": "읽다", "abrir": "열다", "cerrar": "닫다", "empezar": "시작하다", "comenzar": "시작하다",
    "terminar": "끝내다", "jugar": "(운동·게임을) 하다", "dormir": "자다", "morir": "죽다",
    "pedir": "요청하다/주문하다", "servir": "제공하다/쓸모가 있다", "sentir": "느끼다",
    "preferir": "선호하다", "repetir": "반복하다", "traer": "가져오다", "oír": "듣다",
    "caer": "떨어지다", "conducir": "운전하다", "traducir": "번역하다", "construir": "짓다",
    "elegir": "고르다", "corregir": "고치다", "proteger": "보호하다", "cocinar": "요리하다",
    "estudiar": "공부하다", "trabajar": "일하다", "viajar": "여행하다", "comprar": "사다",
    "vender": "팔다", "pagar": "지불하다", "buscar": "찾다", "tocar": "만지다/연주하다",
    "sacar": "꺼내다", "practicar": "연습하다", "explicar": "설명하다", "almorzar": "점심을 먹다",
    "cenar": "저녁을 먹다", "desayunar": "아침을 먹다", "recordar": "기억하다",
    "contar": "세다/이야기하다", "costar": "(비용이) 들다", "mostrar": "보여 주다",
    "probar": "맛보다/시험해 보다", "soñar": "꿈꾸다", "entender": "이해하다", "perder": "잃다",
    "mover": "움직이다", "aprender": "배우다", "comprender": "이해하다", "correr": "달리다",
    "creer": "믿다", "deber": "~해야 하다", "responder": "대답하다", "romper": "깨뜨리다",
    "subir": "올라가다", "decidir": "결정하다", "recibir": "받다", "permitir": "허락하다",
    "ayudar": "돕다", "escuchar": "듣다", "mirar": "보다", "esperar": "기다리다/바라다",
    "necesitar": "필요하다", "usar": "사용하다", "cantar": "노래하다", "bailar": "춤추다",
    "nadar": "수영하다", "caminar": "걷다", "andar": "걷다", "enseñar": "가르치다",
    "preguntar": "묻다", "contestar": "대답하다", "olvidar": "잊다", "cambiar": "바꾸다",
    "lavar": "씻다", "limpiar": "청소하다", "cruzar": "건너다", "reír": "웃다",
    "sonreír": "미소 짓다", "valer": "가치가 있다", "caber": "(공간에) 들어가다",
    "conseguir": "얻다", "mentir": "거짓말하다", "medir": "재다", "oler": "냄새를 맡다",
    "obtener": "얻다", "mantener": "유지하다", "proponer": "제안하다", "suponer": "가정하다",
    "devolver": "돌려주다", "resolver": "해결하다", "describir": "묘사하다",
    "descubrir": "발견하다", "cubrir": "덮다", "llover": None,  # impersonal: filtered out below
}
VERBS = {v: g for v, g in VERBS.items() if g}

# Tense weights: what learners need most gets sampled most.
TENSE_WEIGHTS = {
    ("indicativo", "presente"): 5,
    ("indicativo", "pretérito-perfecto-simple"): 5,
    ("indicativo", "pretérito-imperfecto"): 3,
    ("subjuntivo", "presente"): 5,
    ("indicativo", "futuro"): 3,
    ("condicional", "presente"): 3,
    ("indicativo", "pretérito-perfecto-compuesto"): 2,
    ("subjuntivo", "pretérito-imperfecto-1"): 2,
    ("imperativo", "afirmativo"): 2,
    ("indicativo", "pretérito-pluscuamperfecto"): 1,
    ("subjuntivo", "pretérito-perfecto"): 1,
}
PERSON_WEIGHTS = {"yo": 5, "tú": 5, "él": 4, "nosotros": 4, "ellos": 4, "vosotros": 1}

# Which tense a learner (or a small model) typically confuses with which.
CONFUSABLE = {
    ("subjuntivo", "presente"): [("indicativo", "presente"), ("indicativo", "futuro")],
    ("indicativo", "presente"): [("subjuntivo", "presente")],
    ("condicional", "presente"): [("indicativo", "pretérito-imperfecto"), ("subjuntivo", "pretérito-imperfecto-1")],
    ("indicativo", "pretérito-imperfecto"): [("indicativo", "pretérito-perfecto-simple"), ("condicional", "presente")],
    ("indicativo", "pretérito-perfecto-simple"): [("indicativo", "pretérito-imperfecto"), ("indicativo", "presente")],
    ("indicativo", "futuro"): [("indicativo", "presente"), ("condicional", "presente")],
    ("subjuntivo", "pretérito-imperfecto-1"): [("condicional", "presente"), ("indicativo", "pretérito-imperfecto")],
    ("imperativo", "afirmativo"): [("indicativo", "presente")],
}

SHORT_PROMPTS = [
    "{v}의 {t}, {p} 형태는? 답만 한 단어로 써 주세요.",
    "{v} {t} {p} 형태만 알려 주세요.",
    "{v}의 {t}에서 {p} 형태? 답만요.",
]
ANSWER_PROMPTS = [
    "{v}의 {t}, {p} 형태가 뭐예요?",
    "{v}를 {t} 형태로 바꾸면 {p}는 어떻게 돼요?",
    "{p}가 주어일 때 {v}의 {t} 형태 알려 주세요.",
    "{v} {t} {p} 형태가 헷갈려요. 뭐가 맞아요?",
]
TABLE_PROMPTS = [
    "{v} {t} 변화표 보여 주세요.",
    "{v}의 {t} 동사 변화 전부 알려 주세요.",
    "{v} {t} 인칭별로 정리해 주세요.",
]
VERIFY_PROMPTS = [
    "{v}의 {t}, {p} 형태가 '{x}' 맞나요?",
    "{v} {t} {p}는 {x} 맞죠?",
    "{xq} {v}의 {t} {p} 형태 맞아요?",
]


def eval_triples():
    """(verb, tense_name, person) combos asked in the eval set — excluded from training."""
    found = set()
    for line in (OUT_DIR.parent.parent / "eval" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line) if line.strip() else None
        if item and item["category"] == "conjugation":
            verb = item["prompt"].split("의 ")[0]
            for key, name in TENSES.items():
                if name.split("(")[0] in item["prompt"]:
                    for p in PERSON_KO:
                        if f", {p} 형태" in item["prompt"]:
                            found.add((verb, key, p))
    return found


def pick(rng, weights):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def main():
    rng = random.Random(SEED)
    known = {}
    for v in VERBS:
        try:
            known[v] = conjugate(v)
        except UnknownVerb as e:
            print("skipping", e)
    verbs = list(known)
    banned = eval_triples()
    print(f"{len(verbs)} verbs, {len(banned)} eval combos excluded")

    def sample_triple():
        while True:
            v, key = rng.choice(verbs), pick(rng, TENSE_WEIGHTS)
            table = known[v][key]
            p = pick(rng, {p: w for p, w in PERSON_WEIGHTS.items() if p in table})
            if (v, key, p) not in banned:
                return v, key, p, table[p]

    rows, seen = [], set()

    def add(ex, dedup_key):
        if dedup_key not in seen:
            seen.add(dedup_key)
            rows.append(ex)

    for _ in range(N_SHORT):
        v, key, p, f = sample_triple()
        user = rng.choice(SHORT_PROMPTS).format(v=v, t=TENSES[key], p=p)
        add(example(user, f, kind="conj_short", verb=v), ("s", v, key, p))

    for _ in range(N_ANSWER):
        v, key, p, f = sample_triple()
        user = rng.choice(ANSWER_PROMPTS).format(v=v, t=TENSES[key], p=PERSON_KO[p])
        answer = f"{v}({VERBS[v]})의 {TENSES[key]}, {PERSON_KO[p]} 형태는 **{f}**{copula(f)}."
        if key == ("subjuntivo", "pretérito-imperfecto-1"):
            alt = known[v][("subjuntivo", "pretérito-imperfecto-2")][p]
            answer += f"\n\n-se 형태인 **{alt}**도 같은 뜻으로 쓸 수 있지만, 일상 회화에서는 -ra 형태가 더 흔해요."
        if p == "vosotros":
            answer += "\n\n참고로 vosotros는 스페인에서만 쓰고, 중남미에서는 ustedes(= ellos 형태)를 써요."
        add(example(user, answer, kind="conj_answer", verb=v), ("a", v, key, p))

    for _ in range(N_TABLE):
        v, key = rng.choice(verbs), pick(rng, TENSE_WEIGHTS)
        table = known[v][key]
        user = rng.choice(TABLE_PROMPTS).format(v=v, t=TENSES[key])
        lines = [f"**{v}**({VERBS[v]}) — {TENSES[key]}", "", "| 인칭 | 형태 |", "|---|---|"]
        lines += [f"| {PERSON_KO[p]} | {table[p]} |" for p in PERSON_KO if p in table]
        if key == ("imperativo", "afirmativo"):
            lines += ["", "명령법에는 yo 형태가 없어요. usted/ustedes 명령은 접속법 현재 형태와 같아요."]
        else:
            lines += ["", "usted는 él과, ustedes는 ellos와 같은 형태를 써요. vosotros는 스페인에서만 써요."]
        add(example(user, "\n".join(lines), kind="conj_table", verb=v), ("t", v, key))

    for i in range(N_VERIFY):
        v, key, p, f = sample_triple()
        is_correct = i % 3 == 0  # 1/3 true claims, 2/3 false: the model must learn to say "no"
        wrong_key = None
        if not is_correct:
            options = [k for k in CONFUSABLE.get(key, []) if p in known[v][k] and known[v][k][p] != f]
            if not options:
                continue
            wrong_key = rng.choice(options)
        claim = f if is_correct else known[v][wrong_key][p]
        user = rng.choice(VERIFY_PROMPTS).format(v=v, t=TENSES[key], p=p, x=claim, xq=josa(f"'{claim}'", "이/가"))
        if is_correct:
            answer = f"네, 맞아요! {v}의 {TENSES[key]}, {PERSON_KO[p]} 형태는 **{f}**{copula(f)}."
        else:
            answer = (
                f"아니요, {josa(f'**{claim}**', '은/는')} {v}의 **{TENSES[wrong_key]}** 형태예요.\n\n"
                f"{TENSES[key]}, {PERSON_KO[p]} 형태는 **{f}**{copula(f)}."
            )
        add(example(user, answer, kind="conj_verify", verb=v, correct_claim=is_correct), ("v", v, key, p, claim))

    rng.shuffle(rows)
    write_jsonl(rows, OUT_DIR / "sft_conjugation.jsonl")


if __name__ == "__main__":
    main()
