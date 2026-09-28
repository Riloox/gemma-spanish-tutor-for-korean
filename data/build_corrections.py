"""Build correction SFT examples by *injecting known errors into correct sentences*.

Because we write the correct sentence and choose the error ourselves, we know the
exact fix and the rule behind it. The labels are right by construction, with no
teacher model to trust. Error types map to the pain-point research:

  article_gender   la problema / el foto / la agua     (Korean has no gender)
  agreement        una casa blanco / mi hermana es alto
  ser_estar        soy cansado / soy en Seúl / estoy estudiante / estoy de Incheon / la fiesta está en…
  subjunctive      espero que vienes / cuando terminaré
  gustar           yo gusto el café / me gusta las películas / yo encanto… / me duele los pies
  si_clause        si tengo un coche, viajaría / si tendría… / si lloverá mañana
  personal_a       veo mi madre / espero a el tren
  past_tense       ayer iba al cine / cuando era niño jugué

Both directions of a contrast get examples: estar is sometimes the fix and sometimes the
error, so the model can't learn "estar = correct".

Every item is rendered with its error. A ~15% slice is *also* rendered as the correct
sentence ("네, 맞아요"), so each "correct" example has its broken twin in the data and
the model has to look at the exact word that differs. v2 had 30% correct examples on
separate items and learned "fluent-looking → 👍", passing wrong sentences in the eval.
Step 1 showed models inventing errors in correct sentences, so the slice stays nonzero.

Single-sentence answers open with the check (rule + the form the learner used, ✓/✗) and
only then give the verdict. See check_lines().

    uv run python -m data.build_corrections
"""

import random
from dataclasses import dataclass

from data.common import OUT_DIR, copula, eval_sentences, example, is_contaminated, josa, write_jsonl
from data.conjugator import form

SEED = 7
CORRECT_SHARE = 0.15  # of items per type that also get a "correct as-is" example
PER_TYPE = {
    "article_gender": 260, "agreement": 320, "ser_estar": 360, "subjunctive": 360,
    "gustar": 160, "si_clause": 200, "personal_a": 130, "past_tense": 260,
}
N_MULTI = 260

GENERAL_NOTE = {
    "article_gender": "한국어에는 명사의 성(性)이 없어서 헷갈리기 쉬워요. 새 단어는 꼭 관사와 함께 외우세요.",
    "agreement": "한국어 형용사는 꾸미는 말에 따라 모양이 바뀌지 않아서, 이 부분을 놓치기 쉬워요.",
    "subjunctive": "한국어에는 이런 서법(mood) 구분이 없어서 어렵게 느껴지는 게 당연해요. '바람·의심·감정·판단이 담긴 que절 → 접속법'부터 기억하세요.",
}


@dataclass
class Item:
    type: str
    tmpl: str          # sentence with {X} where the error goes
    right: str
    wrong: str
    ko: str            # Korean translation of the correct sentence
    rule: str          # full explanation
    rule_short: str    # one line, for multi-error feedback
    ok_note: str       # what the learner did right (for correct sentences)

    def sentence(self, x):
        return self.tmpl.replace("{X}", x)

    @property
    def correct(self):
        return self.sentence(self.right)

    @property
    def incorrect(self):
        return self.sentence(self.wrong)


def cap(s):
    return s[0].upper() + s[1:]


# ---------------------------------------------------------------- article gender

NOUNS_WHERE = [  # (noun, gender, Korean) used in "¿Dónde está el/la ___?"
    ("mapa", "m", "지도"), ("sofá", "m", "소파"), ("coche", "m", "자동차"), ("libro", "m", "책"),
    ("paraguas", "m", "우산"), ("hotel", "m", "호텔"), ("tren", "m", "기차"), ("restaurante", "m", "식당"),
    ("foto", "f", "사진"), ("moto", "f", "오토바이"), ("radio", "f", "라디오"), ("llave", "f", "열쇠"),
    ("leche", "f", "우유"), ("mochila", "f", "배낭"), ("estación", "f", "역"), ("universidad", "f", "대학교"),
    ("teléfono", "m", "전화기"), ("pasaporte", "m", "여권"), ("bolso", "m", "가방"), ("reloj", "m", "시계"),
    ("computadora", "f", "컴퓨터"), ("cámara", "f", "카메라"), ("farmacia", "f", "약국"), ("habitación", "f", "방"),
    ("agua", "f*", "물"), ("aula", "f*", "교실"),
]
NOUNS_IMPORTANT = [  # used in "El/La ___ es muy importante."
    ("problema", "m", "문제"), ("tema", "m", "주제"), ("sistema", "m", "시스템"), ("programa", "m", "프로그램"),
    ("idioma", "m", "언어"), ("clima", "m", "기후"), ("examen", "m", "시험"), ("método", "m", "방법"),
    ("lección", "f", "이번 과"), ("clase", "f", "수업"), ("salud", "f", "건강"), ("opinión", "f", "의견"),
    ("verdad", "f", "진실"), ("costumbre", "f", "습관"),
    ("mensaje", "m", "메시지"), ("viaje", "m", "여행"), ("educación", "f", "교육"), ("libertad", "f", "자유"),
    ("paz", "f", "평화"), ("familia", "f", "가족"),
]
NEEDED = {"mapa", "paraguas", "coche", "libro", "llave", "mochila", "foto", "teléfono", "pasaporte", "bolso",
          "reloj", "computadora", "cámara", "habitación", "hotel"}  # countable things you'd "need"; not leche, not places
SHORTENED = {"foto": "fotografía", "moto": "motocicleta", "radio": "radiodifusión"}


def gender_rule(noun, g, art):
    if g == "f*":
        return (f"'{noun}'는 여성 명사지만, 강세가 있는 a-로 시작해서 단수일 때는 발음 때문에 **el**을 써요. "
                f"복수는 다시 las {noun}s이고, 꾸미는 형용사도 여성형이에요 (el {noun} fría).")
    base = f"{josa(f'{noun}', '은/는')} {'남성' if g == 'm' else '여성'} 명사라서 {josa(f'**{art}**', '을/를')} 써요."
    if g == "m" and noun.endswith("ma"):
        return base + " -ma로 끝나는 그리스어 계열 단어(problema, tema, sistema, programa, idioma, clima)는 -a로 끝나도 남성이에요."
    if g == "m" and noun.endswith(("a", "á")):
        return base + " -a로 끝나지만 남성 명사인 예외예요."
    if noun in SHORTENED:
        return base + f" {SHORTENED[noun]}의 줄임말이라서 -o로 끝나도 여성이에요."
    if noun.endswith(("ción", "sión", "dad", "tad", "tud")):
        return base + " -ción, -sión, -dad, -tud로 끝나는 명사는 여성이에요."
    if noun.endswith("o"):
        return base + " -o로 끝나는 명사는 대부분 남성이에요."
    if noun.endswith("a"):
        return base + " -a로 끝나는 명사는 대부분 여성이에요."
    return base + " -e나 자음으로 끝나는 명사는 성을 따로 외워야 해요."


def gen_article_gender():
    items = []
    for noun, g, ko in NOUNS_WHERE:
        right = "la" if g == "f" else "el"
        wrong = "el" if right == "la" else "la"
        rule = gender_rule(noun, g, right)
        items.append(Item(
            "article_gender", "¿Dónde está {X} " + noun + "?", right, wrong,
            f"{josa(ko, '은/는')} 어디 있어요?", rule + " " + GENERAL_NOTE["article_gender"],
            f"{noun}{'는 el을 쓰는 여성 명사' if g == 'f*' else ('는 남성' if g == 'm' else '는 여성')} → {right} {noun}",
            f"'{noun}' 앞에 {josa(right, '을/를')} 정확하게 썼어요.",
        ))
    for noun, g, ko in NOUNS_WHERE:
        if noun not in NEEDED:
            continue
        right, wrong = ("una", "un") if g == "f" else ("un", "una")
        items.append(Item(
            "article_gender", "Necesito {X} " + noun + ".", right, wrong,
            f"{josa(ko, '이/가')} 필요해요.", gender_rule(noun, g, right) + " " + GENERAL_NOTE["article_gender"],
            f"{noun}는 {'남성' if g == 'm' else '여성'} 명사 → {right} {noun}",
            f"'{noun}'의 성에 맞게 {josa(right, '을/를')} 썼어요.",
        ))
    for noun, g, ko in NOUNS_IMPORTANT:
        right, wrong = ("La", "El") if g == "f" else ("El", "La")
        items.append(Item(
            "article_gender", "{X} " + noun + " es muy importante.", right, wrong,
            f"{josa(ko, '은/는')} 아주 중요해요.", gender_rule(noun, g, right.lower()) + " " + GENERAL_NOTE["article_gender"],
            f"{noun}는 {'남성' if g == 'm' else '여성'} 명사 → {right.lower()} {noun}",
            f"'{noun}'의 성을 정확히 알고 {josa(right.lower(), '을/를')} 썼어요.",
        ))
    return items


# ---------------------------------------------------------------- agreement

ADJ = {  # masc, fem, masc pl, fem pl, Korean modifier, Korean predicate (None = don't use as predicate)
    "pequeño": ("pequeño", "pequeña", "pequeños", "pequeñas", "작은", "아주 작아요"),
    "bonito": ("bonito", "bonita", "bonitos", "bonitas", "예쁜", "아주 예뻐요"),
    "viejo": ("viejo", "vieja", "viejos", "viejas", "오래된", "아주 오래됐어요"),
    "caro": ("caro", "cara", "caros", "caras", "비싼", "아주 비싸요"),
    "barato": ("barato", "barata", "baratos", "baratas", "싼", "아주 싸요"),
    "nuevo": ("nuevo", "nueva", "nuevos", "nuevas", "새", None),
    "blanco": ("blanco", "blanca", "blancos", "blancas", "하얀", None),
    "negro": ("negro", "negra", "negros", "negras", "검은", None),
    "rojo": ("rojo", "roja", "rojos", "rojas", "빨간", None),
}
OBJECTS = [  # noun, gender, plural, Korean, allowed adjectives
    ("casa", "f", "casas", "집", ["pequeño", "bonito", "viejo", "caro", "nuevo", "blanco"]),
    ("coche", "m", "coches", "자동차", ["pequeño", "viejo", "caro", "barato", "nuevo", "rojo", "negro", "blanco"]),
    ("libro", "m", "libros", "책", ["viejo", "caro", "barato", "nuevo", "pequeño"]),
    ("mesa", "f", "mesas", "테이블", ["pequeño", "viejo", "nuevo", "blanco", "negro", "bonito"]),
    ("bicicleta", "f", "bicicletas", "자전거", ["viejo", "caro", "barato", "nuevo", "rojo", "negro"]),
    ("camisa", "f", "camisas", "셔츠", ["caro", "barato", "nuevo", "bonito", "blanco", "negro", "rojo"]),
    ("mochila", "f", "mochilas", "배낭", ["pequeño", "bonito", "nuevo", "rojo", "negro"]),
    ("mapa", "m", "mapas", "지도", ["viejo", "nuevo", "pequeño"]),
    ("moto", "f", "motos", "오토바이", ["viejo", "caro", "nuevo", "rojo", "negro"]),
    ("perro", "m", "perros", "강아지", ["pequeño", "bonito", "blanco", "negro"]),
    ("gato", "m", "gatos", "고양이", ["pequeño", "bonito", "blanco", "negro"]),
]
PEOPLE = [  # Spanish, gender, Korean
    ("mi hermano", "m", "제 남동생"), ("mi hermana", "f", "제 여동생"), ("mi padre", "m", "제 아버지"),
    ("mi madre", "f", "제 어머니"), ("mi amigo", "m", "제 친구"), ("mi amiga", "f", "제 친구"),
    ("mi novio", "m", "제 남자친구"), ("mi novia", "f", "제 여자친구"), ("mi profesora", "f", "제 선생님"),
    ("mi profesor", "m", "제 선생님"),
]
TRAITS = {  # masc, fem, Korean predicate
    "alto": ("alto", "alta", "키가 아주 커요"), "bajo": ("bajo", "baja", "키가 아주 작아요"),
    "simpático": ("simpático", "simpática", "아주 친절해요"), "divertido": ("divertido", "divertida", "아주 재미있어요"),
    "trabajador": ("trabajador", "trabajadora", "아주 부지런해요"), "tímido": ("tímido", "tímida", "수줍음이 아주 많아요"),
}


def gen_agreement():
    items = []
    for noun, g, pl, ko, adjs in OBJECTS:
        art = "una" if g == "f" else "un"
        for a in adjs:
            m, f, mp, fp, mod, pred = ADJ[a]
            right, wrong = (f, m) if g == "f" else (m, f)
            items.append(Item(
                "agreement", f"Tengo {art} {noun} {{X}}.", right, wrong,
                f"저는 {mod} {josa(ko, '이/가')} 있어요.",
                f"형용사는 꾸미는 명사의 성에 맞춰야 해요. {josa(noun, '은/는')} {'여성' if g == 'f' else '남성'} 명사라서 "
                f"{josa(wrong, '이/가')} 아니라 **{right}**{copula(right)}. " + GENERAL_NOTE["agreement"],
                f"{noun}({'여성' if g == 'f' else '남성'}) → {right}",
                f"{noun}의 성에 맞게 형용사를 {josa(right, '으로/로')} 잘 맞췄어요.",
            ))
            if pred:
                right_p, wrong_p = (fp, f) if g == "f" else (mp, m)
                art_p = "Las" if g == "f" else "Los"
                items.append(Item(
                    "agreement", f"{art_p} {pl} son muy {{X}}.", right_p, wrong_p,
                    f"그 {josa(ko + '들', '은/는')} {pred}.",
                    f"주어({art_p.lower()} {pl})가 복수라서 형용사도 복수형이 돼요: {wrong_p} → **{right_p}**. "
                    f"성({'여성' if g == 'f' else '남성'})과 수(복수)를 둘 다 맞춰야 해요. " + GENERAL_NOTE["agreement"],
                    f"복수 주어 → 형용사도 복수 {right_p}",
                    f"복수 주어에 맞춰 {josa(right_p, '으로/로')} 잘 썼어요.",
                ))
    for who, g, ko in PEOPLE:
        for t, (m, f, pred) in TRAITS.items():
            right, wrong = (f, m) if g == "f" else (m, f)
            extra = " -dor로 끝나는 형용사는 여성형이 -dora가 돼요." if t == "trabajador" else ""
            items.append(Item(
                "agreement", f"{cap(who)} es muy {{X}}.", right, wrong,
                f"{josa(ko, '은/는')} {pred}.",
                f"형용사는 설명하는 사람의 성에 맞춰요. {who}는 {'여성' if g == 'f' else '남성'}이라서 "
                f"{josa(wrong, '이/가')} 아니라 {josa(f'**{right}**', '을/를')} 써요.{extra} " + GENERAL_NOTE["agreement"],
                f"{who}({'여성' if g == 'f' else '남성'}) → {right}",
                f"{who}에 맞게 {josa(right, '을/를')} 잘 썼어요.",
            ))
    return items


# ---------------------------------------------------------------- ser / estar

STATES = {  # masc sg, Korean predicate
    "cansado": "아주 피곤해요", "ocupado": "아주 바빠요", "contento": "아주 기뻐요", "nervioso": "아주 긴장돼요",
    "enfermo": "많이 아파요", "triste": "아주 슬퍼요", "aburrido": "아주 심심해요",
}
STATE_SUBJECTS = [  # template, person, gender, number, Korean subject
    ("Hoy {V} muy {A}.", "yo", "m", "s", "저는 오늘"),
    ("Hoy {V} muy {A}.", "nosotros", "m", "p", "우리는 오늘"),
    ("Mi madre {V} muy {A} hoy.", "él", "f", "s", "제 어머니는 오늘"),
    ("Mi hermano {V} muy {A} hoy.", "él", "m", "s", "제 남동생은 오늘"),
    ("Mis amigas {V} muy {A} hoy.", "ellos", "f", "p", "제 친구들은 오늘"),
]
PLACES = [("Seúl", "서울"), ("Busan", "부산"), ("Madrid", "마드리드"), ("México", "멕시코"), ("casa", "집"),
          ("la oficina", "사무실"), ("el hospital", "병원"), ("la biblioteca", "도서관"), ("el aeropuerto", "공항")]
PLACE_SUBJECTS = [
    ("Ahora {V} en {P}.", "yo", "저는 지금 {P}에 있어요."),
    ("Ahora {V} en {P}.", "nosotros", "우리는 지금 {P}에 있어요."),
    ("Mi padre {V} en {P} ahora.", "él", "제 아버지는 지금 {P}에 계세요."),
    ("Mis amigos {V} en {P} ahora.", "ellos", "제 친구들은 지금 {P}에 있어요."),
]
JOBS = [  # masc, fem, masc pl, Korean
    ("estudiante", "estudiante", "estudiantes", "학생"), ("profesor", "profesora", "profesores", "선생님"),
    ("médico", "médica", "médicos", "의사"), ("ingeniero", "ingeniera", "ingenieros", "엔지니어"),
    ("enfermero", "enfermera", "enfermeros", "간호사"), ("coreano", "coreana", "coreanos", "한국 사람"),
    ("cocinero", "cocinera", "cocineros", "요리사"),
]
JOB_SUBJECTS = [  # template, person, which job form (None = either), Korean subject
    ("{V} {J}.", "yo", None, "저는"), ("Yo {V} {J}.", "yo", "m", "저는"),  # learners often keep the pronoun
    ("Mi hermana {V} {J}.", "él", "f", "제 여동생은"),
    ("Mi hermano {V} {J}.", "él", "m", "제 남동생은"), ("Mis amigos {V} {J}.", "ellos", "mp", "제 친구들은 모두"),
    ("Mi padre {V} {J}.", "él", "m", "제 아버지는"), ("Mi madre {V} {J}.", "él", "f", "제 어머니는"),
    ("Nosotros {V} {J}.", "nosotros", "mp", "우리는 모두"),
]
ORIGINS = [("Incheon", "인천"), ("Gwangju", "광주"), ("Jeju", "제주"), ("Madrid", "마드리드"), ("Lima", "리마"),
           ("Bogotá", "보고타")]
ORIGIN_SUBJECTS = [("{V} de {C}.", "yo", "저는 {C} 출신이에요."), ("Mi novia {V} de {C}.", "él", "제 여자친구는 {C} 출신이에요."),
                   ("Mis padres {V} de {C}.", "ellos", "제 부모님은 {C} 출신이세요.")]
EVENTS = [("La fiesta", "파티", "파티는"), ("El concierto", "콘서트", "콘서트는"), ("La reunión", "회의", "회의는"),
          ("La boda", "결혼식", "결혼식은"), ("El examen", "시험", "시험은")]
EVENT_PLACES = [("en mi casa", "저희 집에서"), ("en el hotel", "호텔에서"), ("en la universidad", "대학교에서"),
                ("en el centro", "시내에서")]


def agree(adj_m, g, n):
    base = adj_m[:-1] + ("a" if g == "f" else "o") if adj_m.endswith("o") else adj_m
    return base + ("s" if n == "p" else "")


def gen_ser_estar():
    items = []
    for adj, ko_pred in STATES.items():
        for tmpl, p, g, n, ko_subj in STATE_SUBJECTS:
            a = agree(adj, g, n)
            right, wrong = form("estar", "indicativo", "presente", p), form("ser", "indicativo", "presente", p)
            rule = ("피곤함·바쁨·기쁨처럼 '지금 이 순간의 상태'는 **estar**를 써요. ser는 국적·직업·성격처럼 "
                    "그 사람을 설명하는 특징에 써요.")
            if adj == "aburrido":
                rule += " 특히 aburrido는 estar와 쓰면 '심심하다', ser와 쓰면 '지루한 사람이다'라는 뜻이라 의미가 완전히 달라져요."
            items.append(Item(
                "ser_estar", tmpl.replace("{V}", "{X}").replace("{A}", a).replace("Hoy {X}", "Hoy {X}"),
                right, wrong, f"{ko_subj} {ko_pred}.", rule,
                f"지금의 상태({a}) → estar", "지금의 상태라서 estar를 잘 썼어요.",
            ))
    for place, ko_place in PLACES:
        for tmpl, p, ko in PLACE_SUBJECTS:
            right, wrong = form("estar", "indicativo", "presente", p), form("ser", "indicativo", "presente", p)
            items.append(Item(
                "ser_estar", tmpl.replace("{V}", "{X}").replace("{P}", place), right, wrong, ko.replace("{P}", ko_place),
                "사람이나 물건이 '어디에 있다'는 위치는 항상 **estar**를 써요. 한국어의 '있다'와 비슷하다고 생각하면 쉬워요. "
                "(단, 파티나 회의 같은 행사가 열리는 장소는 ser를 써요: La fiesta es en mi casa.)",
                "위치 → estar", "위치를 말할 때 estar를 잘 썼어요.",
            ))
    for m, f, mp, ko_job in JOBS:
        for tmpl, p, which, ko_subj in JOB_SUBJECTS:
            job = {"m": m, "f": f, "mp": mp, None: random.Random(m).choice([m, f])}[which]
            right, wrong = form("ser", "indicativo", "presente", p), form("estar", "indicativo", "presente", p)
            x_right, x_wrong = (cap(right), cap(wrong)) if tmpl.startswith("{V}") else (right, wrong)
            items.append(Item(
                "ser_estar", tmpl.replace("{V}", "{X}").replace("{J}", job), x_right, x_wrong,
                f"{ko_subj} {ko_job}{copula(ko_job)}.",
                "직업·국적·신분처럼 '그 사람이 누구인지'는 **ser**를 써요. estar는 일시적인 상태나 위치에 써요. "
                "참고로 직업 앞에는 보통 관사(un/una)를 붙이지 않아요: Soy estudiante.",
                "직업·국적 → ser", "직업이나 국적을 말할 때 ser를 잘 썼어요.",
            ))
    for city, ko_city in ORIGINS:
        for tmpl, p, ko in ORIGIN_SUBJECTS:
            right, wrong = form("ser", "indicativo", "presente", p), form("estar", "indicativo", "presente", p)
            x_right, x_wrong = (cap(right), cap(wrong)) if tmpl.startswith("{V}") else (right, wrong)
            items.append(Item(
                "ser_estar", tmpl.replace("{V}", "{X}").replace("{C}", city), x_right, x_wrong, ko.replace("{C}", ko_city),
                "출신(어디에서 왔는지)은 **ser de**로 말해요. 출신은 바뀌지 않는 그 사람의 정보라서 ser예요. "
                f"지금 있는 위치라면 estar en을 써요: Estoy en {city}.",
                "출신 → ser de", "출신을 말할 때 ser de를 잘 썼어요.",
            ))
    for event, ko_noun, ko_event in EVENTS:
        for place, ko_place in EVENT_PLACES:
            items.append(Item(
                "ser_estar", f"{event} {{X}} {place}.", "es", "está", f"{ko_event} {ko_place} 열려요.",
                "사람이나 물건의 위치는 estar지만, 파티·콘서트·회의처럼 **행사가 열리는 장소**는 **ser**를 써요. "
                f"{ko_event} 어딘가에 '있는' 게 아니라 '열리는' 것이라서 es예요. "
                "(비교: Mi casa está cerca del centro. 집의 위치는 estar)",
                "행사가 열리는 장소 → ser", "행사가 열리는 장소에 ser를 잘 썼어요.",
            ))
    return items


# ---------------------------------------------------------------- subjunctive

CLAUSES = [  # verb, complement, Korean forms: 기 / 는 게 / 다고 / 면
    ("venir", "a la fiesta", "파티에 오기", "파티에 오는 게", "파티에 온다고", "파티에 오면"),
    ("tener", "tiempo mañana", "내일 시간이 있기", "내일 시간이 있는 게", "내일 시간이 있다고", "내일 시간이 있으면"),
    ("llegar", "temprano", "일찍 도착하기", "일찍 도착하는 게", "일찍 도착한다고", "일찍 도착하면"),
    ("estudiar", "mucho", "공부를 많이 하기", "공부를 많이 하는 게", "공부를 많이 한다고", "공부를 많이 하면"),
    ("dormir", "bien", "잘 자기", "잘 자는 게", "잘 잔다고", "잘 자면"),
    ("decir", "la verdad", "사실대로 말하기", "사실대로 말하는 게", "사실대로 말한다고", "사실대로 말하면"),
    ("encontrar", "trabajo", "일자리를 찾기", "일자리를 찾는 게", "일자리를 찾는다고", "일자리를 찾으면"),
    ("hacer", "la tarea", "숙제를 하기", "숙제를 하는 게", "숙제를 한다고", "숙제를 하면"),
    ("volver", "pronto", "곧 돌아오기", "곧 돌아오는 게", "곧 돌아온다고", "곧 돌아오면"),
    ("salir", "de casa temprano", "집에서 일찍 나가기", "집에서 일찍 나가는 게", "집에서 일찍 나간다고", "집에서 일찍 나가면"),
]
SUBJ_SUBJECTS = [("tú", "tú", "네가"), ("mi hermano", "él", "제 남동생이"),
                 ("ustedes", "ellos", "여러분이"), ("mis hijos", "ellos", "우리 아이들이")]
TRIGGERS = [  # Spanish start, Korean builder, rule
    ("Espero que", lambda s, k: f"{s} {k[0]}를 바라요.",
     "esperar que(~하기를 바라다)처럼 **바람이나 소망**을 나타내는 표현 뒤의 que절에는 접속법을 써요."),
    ("Quiero que", lambda s, k: f"{s} {k[0]}를 원해요.",
     "querer que(~하기를 원하다)처럼 **다른 사람에게 바라는 것**을 말할 때 que절에는 접속법을 써요."),
    ("Es importante que", lambda s, k: f"{s} {k[1]} 중요해요.",
     "es importante que, es necesario que처럼 **판단·필요**를 나타내는 표현 뒤에는 접속법을 써요."),
    ("No creo que", lambda s, k: f"{s} {k[2]} 생각하지 않아요.",
     "creer que는 긍정문에서는 직설법(Creo que tiene razón)을 쓰지만, **부정하면(no creo que) 의심**을 나타내서 접속법을 써요."),
    ("Ojalá que", lambda s, k: f"{s} {k[3]} 좋겠어요.",
     "ojalá(que)는 '~하면 좋겠다'는 **소망**이라서 항상 접속법을 써요."),
]
CUANDO = [  # yo-clause, Korean, main clause, Korean main
    ("terminar", "la tarea", "숙제를 끝내면"), ("tener", "tiempo", "시간이 있으면"),
    ("salir", "del trabajo", "퇴근하면"), ("volver", "de Madrid", "마드리드에서 돌아오면"),
    ("saber", "el resultado", "결과를 알게 되면"), ("llegar", "al hotel", "호텔에 도착하면"),
]


def gen_subjunctive():
    items = []
    for verb, comp, *ko_forms in CLAUSES:
        for trig, ko_fn, rule in TRIGGERS:
            for subj_es, p, subj_ko in SUBJ_SUBJECTS:
                right = form(verb, "subjuntivo", "presente", p)
                wrong = form(verb, "indicativo", "presente", p)
                if right == wrong:
                    continue
                items.append(Item(
                    "subjunctive", f"{trig} {subj_es} {{X}} {comp}.", right, wrong,
                    ko_fn(subj_ko, ko_forms),
                    f"{rule} 그래서 직설법 {josa(wrong, '이/가')} 아니라 접속법 {josa(f'**{right}**', '을/를')} 써야 해요. " + GENERAL_NOTE["subjunctive"],
                    f"{trig} + 접속법 → {right}",
                    f"{trig} 뒤에 접속법 {josa(right, '을/를')} 정확하게 썼어요.",
                ))
    for verb, comp, ko in CUANDO:
        right = form(verb, "subjuntivo", "presente", "yo")
        wrong = form(verb, "indicativo", "futuro", "yo")
        items.append(Item(
            "subjunctive", f"Cuando {{X}} {comp}, te llamo.", right, wrong, f"{ko} 전화할게요.",
            "cuando 뒤에 **아직 일어나지 않은 미래의 일**이 오면 미래형이 아니라 **접속법 현재**를 써요. "
            f"그래서 {josa(wrong, '이/가')} 아니라 **{right}**{copula(right)}. "
            "(습관처럼 반복되는 일은 직설법: Cuando llego a casa, siempre ceno.) " + GENERAL_NOTE["subjunctive"],
            f"cuando + 미래의 일 → 접속법 {right}",
            f"cuando 뒤 미래의 일에 접속법 {josa(right, '을/를')} 잘 썼어요.",
        ))
    return items


# ---------------------------------------------------------------- gustar

LIKES_SG = [("el café", "커피"), ("la música", "음악"), ("el fútbol", "축구"), ("la comida coreana", "한국 음식"),
            ("el chocolate", "초콜릿"), ("el invierno", "겨울"), ("la playa", "바다"), ("la pizza", "피자"),
            ("el verano", "여름"), ("el arte", "미술"), ("la fruta", "과일")]
LIKES_INF = [("bailar", "춤추는 것"), ("cocinar", "요리하는 것"), ("leer", "책 읽는 것"), ("cantar", "노래하는 것"),
             ("nadar", "수영하는 것"), ("caminar", "걷는 것")]
PAIN = [("la cabeza", "머리가", False), ("el estómago", "배가", False), ("la espalda", "등이", False),
        ("la garganta", "목이", False), ("los pies", "발이", True), ("los ojos", "눈이", True),
        ("las piernas", "다리가", True), ("los dientes", "이가", True)]
YO_GUSTO_RULE = ("gustar는 '좋아하다'가 아니라 '~이 (나에게) 좋다'는 구조예요. 좋아하는 대상이 주어라서 "
                 "**Me gusta {thing}**({ko} 나에게 좋다)처럼 써요. 'Yo gusto'는 '내가 (누군가의) 마음에 든다'는 뜻이 돼요.")
LIKES_PL = [("las películas", "영화"), ("los libros", "책"), ("los perros", "강아지"), ("las flores", "꽃"),
            ("los idiomas", "외국어"), ("los gatos", "고양이"), ("los deportes", "운동")]


def gen_gustar():
    items = []
    for thing, ko in LIKES_SG:
        items.append(Item(
            "gustar", "{X} mucho " + thing + ".", "Me gusta", "Yo gusto", f"저는 {josa(ko, '을/를')} 정말 좋아해요.",
            YO_GUSTO_RULE.format(thing=thing, ko=josa(ko, "이/가")),
            "gustar는 '~이 나에게 좋다' → Me gusta", "gustar 구문을 정확하게 썼어요.",
        ))
        items.append(Item(
            "gustar", "A mi hermana {X} mucho " + thing + ".", "le gusta", "gusta",
            f"제 여동생은 {josa(ko, '을/를')} 정말 좋아해요.",
            "gustar는 '~이 (누구에게) 좋다'는 구조라서, 좋아하는 사람은 간접목적어 대명사로 나타내요: "
            f"A mi hermana **le gusta** {thing}. le가 빠지면 틀린 문장이 돼요.",
            "누가 좋아하는지 → le gusta", "간접목적어 le까지 정확하게 썼어요.",
        ))
        items.append(Item(
            "gustar", "{X} " + thing + ".", "Me encanta", "Yo encanto", f"저는 {josa(ko, '이/가')} 정말 좋아요.",
            "encantar(아주 좋아하다)도 gustar처럼 '~이 나에게 아주 좋다'는 구조예요. 좋아하는 대상이 주어라서 "
            f"**Me encanta {thing}**처럼 써요. 'Yo encanto'는 '내가 (누군가를) 매혹한다'는 뜻이 돼요.",
            "encantar도 gustar 구조 → Me encanta", "encantar를 gustar처럼 정확하게 썼어요.",
        ))
    for verb, ko in LIKES_INF:
        items.append(Item(
            "gustar", "{X} mucho " + verb + ".", "Me gusta", "Yo gusto", f"저는 {josa(ko, '을/를')} 정말 좋아해요.",
            YO_GUSTO_RULE.format(thing=verb, ko=josa(ko, "이/가")) + " 동사원형이 오면 동사는 항상 단수 gusta예요.",
            "gustar + 동사원형 → Me gusta", "gustar 구문을 정확하게 썼어요.",
        ))
        items.append(Item(
            "gustar", "A mi hermano {X} mucho " + verb + ".", "le gusta", "gusta",
            f"제 남동생은 {josa(ko, '을/를')} 정말 좋아해요.",
            "gustar는 '~이 (누구에게) 좋다'는 구조라서, 좋아하는 사람은 간접목적어 대명사로 나타내요: "
            f"A mi hermano **le gusta** {verb}. le가 빠지면 틀린 문장이 돼요.",
            "누가 좋아하는지 → le gusta", "간접목적어 le까지 정확하게 썼어요.",
        ))
    for thing, ko in LIKES_PL:
        items.append(Item(
            "gustar", "A mis padres {X} mucho " + thing + ".", "les gustan", "les gusta",
            f"제 부모님은 {josa(ko, '을/를')} 정말 좋아하세요.",
            f"gustar 문장에서 주어는 좋아하는 대상이에요. 대상({thing})이 복수라서 **gustan**이고, "
            "좋아하는 사람(mis padres)이 복수라서 간접목적어도 **les**예요. 둘 다 맞춰야 해요.",
            f"대상이 복수({thing}) → les gustan", "les와 gustan을 둘 다 정확하게 맞췄어요.",
        ))
    for part, ko, plural in PAIN:
        right, wrong = ("duelen", "duele") if plural else ("duele", "duelen")
        items.append(Item(
            "gustar", "Hoy me {X} " + part + ".", right, wrong, f"오늘 {ko} 아파요.",
            f"doler(아프다)는 gustar처럼 **아픈 곳이 주어**예요. {part}{'가 복수라서' if plural else '가 단수라서'} "
            f"**{right}**{copula(right)}. 아픈 사람은 me/te/le로 나타내요.",
            f"아픈 곳({part}) → {right}", f"아픈 곳에 맞춰 {josa(right, '을/를')} 잘 썼어요.",
        ))
    for thing, ko in LIKES_PL:
        items.append(Item(
            "gustar", "Me {X} mucho " + thing + ".", "gustan", "gusta", f"저는 {josa(ko, '을/를')} 정말 좋아해요.",
            f"gustar 문장에서 주어는 좋아하는 대상이에요. 대상({thing})이 복수라서 동사도 복수형 **gustan**을 써요.",
            f"대상이 복수({thing}) → gustan", "복수 대상에 맞춰 gustan을 잘 썼어요.",
        ))
    return items


# ---------------------------------------------------------------- si clauses

UNREAL = [  # si-verb, complement, main verb, main complement (with {pos}), Korean condition, Korean result
    ("tener", "un coche", "viajar", "por todo el país", "차가 있다면", "전국을 여행할 텐데요"),
    ("vivir", "en España", "hablar", "español todos los días", "스페인에 산다면", "매일 스페인어를 쓸 텐데요"),
    ("poder", "volar", "ir", "a Madrid cada fin de semana", "날 수 있다면", "주말마다 마드리드에 갈 텐데요"),
    ("saber", "cocinar", "hacer", "paella para {pos} amigos", "요리를 할 줄 안다면", "친구들에게 빠에야를 만들어 줄 텐데요"),
    ("estar", "en Seúl", "visitar", "a {pos} abuelos", "서울에 있다면", "할머니 할아버지를 찾아뵐 텐데요"),
    ("tener", "vacaciones", "ir", "a la playa", "휴가가 있다면", "바다에 갈 텐데요"),
    ("hablar", "bien inglés", "trabajar", "en Canadá", "영어를 잘한다면", "캐나다에서 일할 텐데요"),
    ("vivir", "cerca del mar", "nadar", "todos los días", "바닷가 근처에 산다면", "매일 수영할 텐데요"),
    ("tener", "más tiempo", "aprender", "a tocar la guitarra", "시간이 더 있다면", "기타를 배울 텐데요"),
    ("ganar", "la lotería", "comprar", "una casa en la playa", "복권에 당첨된다면", "바닷가에 집을 살 텐데요"),
    ("poder", "elegir", "vivir", "en Barcelona", "고를 수 있다면", "바르셀로나에서 살 텐데요"),
    ("tener", "un perro", "salir", "a caminar todos los días", "강아지가 있다면", "매일 산책하러 나갈 텐데요"),
    ("estudiar", "más", "aprobar", "el examen", "공부를 더 한다면", "시험에 합격할 텐데요"),
    ("saber", "nadar", "ir", "a la piscina con {pos} amigos", "수영을 할 줄 안다면", "친구들이랑 수영장에 갈 텐데요"),
]
REAL = [  # si-verb, complement, main clause, Korean condition, Korean result
    ("tener", "tiempo el sábado", "te llamo", "토요일에 시간이 있으면", "전화할게요"),
    ("terminar", "el trabajo temprano", "voy al cine", "일을 일찍 끝내면", "영화 보러 갈게요"),
    ("llegar", "a tiempo", "te escribo", "제시간에 도착하면", "문자할게요"),
    ("encontrar", "las llaves", "te aviso", "열쇠를 찾으면", "알려 줄게요"),
    ("ganar", "el partido", "invito a todos", "경기에서 이기면", "모두에게 한턱낼게요"),
    ("salir", "temprano de clase", "paso por tu casa", "수업이 일찍 끝나면", "너희 집에 들를게요"),
    ("ver", "a Juan", "le digo la verdad", "후안을 보면", "사실대로 말할게요"),
    ("tener", "hambre", "como algo", "배가 고프면", "뭘 좀 먹을게요"),
    ("volver", "pronto", "cenamos juntos", "일찍 돌아오면", "같이 저녁 먹어요"),
    ("estar", "libre el domingo", "te ayudo con la mudanza", "일요일에 한가하면", "이사하는 거 도와줄게요"),
    ("recibir", "el dinero", "te pago", "돈을 받으면", "갚을게요"),
]
UNREAL_RULE = ("지금 사실이 아닌 일을 가정할 때(~라면 ~할 텐데)는 si 뒤에 **접속법 과거**({right})를, "
               "결과 부분에 **조건법**({main})을 써요.")


def gen_si_clause():
    items = []
    for verb, comp, main_verb, main_comp, ko_if, ko_then in UNREAL:
        for p, pos in (("yo", "mis"), ("nosotros", "nuestros"), ("tú", "tus")):
            right = form(verb, "subjuntivo", "pretérito-imperfecto-1", p)
            main = form(main_verb, "condicional", "presente", p)
            tmpl = f"Si {{X}} {comp}, {main} {main_comp.replace('{pos}', pos)}."
            for wrong, why in (
                (form(verb, "indicativo", "presente", p),
                 "si + 직설법 현재는 실제로 일어날 수 있는 조건(Si tengo tiempo, te llamo)에 써요. 조건법과 섞으면 안 돼요."),
                (form(verb, "condicional", "presente", p),
                 "si 바로 뒤에는 조건법을 쓰지 않아요. 조건법은 결과 부분에만 와요."),
            ):
                items.append(Item(
                    "si_clause", tmpl, right, wrong, f"{ko_if} {ko_then}.",
                    UNREAL_RULE.format(right=right, main=main) + f" 그래서 {josa(wrong, '이/가')} 아니라 **{right}**{copula(right)}. " + why,
                    f"사실이 아닌 가정 → si + 접속법 과거 {right}",
                    "사실이 아닌 가정에 si + 접속법 과거를 정확하게 썼어요.",
                ))
    for verb, comp, main, ko_if, ko_then in REAL:
        right = form(verb, "indicativo", "presente", "yo")
        wrong = form(verb, "indicativo", "futuro", "yo")
        items.append(Item(
            "si_clause", f"Si {{X}} {comp}, {main}.", right, wrong, f"{ko_if} {ko_then}.",
            "실제로 일어날 수 있는 조건(~하면)은 si 뒤에 **직설법 현재**를 써요. 미래의 일이라도 si 바로 뒤에는 "
            f"미래형을 쓰지 않아서 {josa(wrong, '이/가')} 아니라 **{right}**{copula(right)}. 결과 부분에는 현재나 미래를 써요.",
            f"si + 실제 조건 → 직설법 현재 {right}", "si 뒤에 직설법 현재를 잘 썼어요.",
        ))
    return items


# ---------------------------------------------------------------- personal a

PERSONS_A = [  # Spanish, Korean, honorific?
    ("mi madre", "어머니", True), ("mi abuela", "할머니", True), ("mi profesor", "선생님", True),
    ("mis amigos", "친구들", False), ("mi novia", "여자친구", False), ("Juan", "후안", False),
    ("mi hermano", "남동생", False), ("mis abuelos", "조부모님", True), ("mi tía", "이모", True),
    ("María", "마리아", False), ("mi vecina", "이웃", False),
    ("mi padre", "아버지", True), ("mis hijos", "아이들", False), ("Ana", "아나", False),
    ("mi jefe", "상사", True), ("mis primos", "사촌들", False),
]
A_FRAMES = [  # Spanish template, Korean (plain), Korean (honorific)
    ("Veo {X} todos los días.", "저는 매일 {K} 봐요.", "저는 매일 {K} 봬요."),
    ("Llamo {X} cada semana.", "저는 매주 {E} 전화해요.", "저는 매주 {H} 전화드려요."),
    ("Visito {X} los domingos.", "저는 일요일마다 {K} 찾아가요.", "저는 일요일마다 {K} 찾아뵈어요."),
    ("Espero {X} en la estación.", "저는 역에서 {K} 기다려요.", "저는 역에서 {K} 기다려요."),
    ("Quiero mucho {X}.", "저는 {K} 많이 사랑해요.", "저는 {K} 많이 사랑해요."),
    ("Busco {X} en el parque.", "저는 공원에서 {K} 찾고 있어요.", "저는 공원에서 {K} 찾고 있어요."),
    ("Invito {X} a cenar.", "저는 {K} 저녁 식사에 초대해요.", "저는 {K} 저녁 식사에 모셔요."),
]
THINGS_A = [  # the other direction: things and places take no a. Template, object, Korean
    ("Veo {X} todos los días.", "la televisión", "저는 매일 텔레비전을 봐요."),
    ("Veo {X} todos los días.", "las noticias", "저는 매일 뉴스를 봐요."),
    ("Busco {X} en el parque.", "mis llaves", "저는 공원에서 열쇠를 찾고 있어요."),
    ("Espero {X} en la estación.", "el tren", "저는 역에서 기차를 기다려요."),
    ("Espero {X} en la parada.", "el autobús", "저는 정류장에서 버스를 기다려요."),
    ("Visito {X} los domingos.", "el museo", "저는 일요일마다 박물관에 가요."),
    ("Visito {X} cada verano.", "Madrid", "저는 여름마다 마드리드를 방문해요."),
    ("Necesito {X} hoy.", "el coche", "저는 오늘 차가 필요해요."),
]


def gen_personal_a():
    items = []
    for who, ko, hon in PERSONS_A:
        for tmpl, ko_plain, ko_hon in A_FRAMES:
            k = (ko_hon if hon else ko_plain).replace("{K}", josa(ko, "을/를")).replace("{E}", ko + "에게").replace("{H}", ko + "께")
            items.append(Item(
                "personal_a", tmpl, f"a {who}", who, k,
                "사람이 직접목적어일 때(누구를 보다·부르다·방문하다)는 앞에 **a**를 붙여요. 이걸 '인칭의 a(a personal)'라고 해요. "
                "사물에는 붙이지 않아요: Veo la televisión. 한국어의 '~을/를'과 달리 사람일 때만 생긴다는 게 핵심이에요.",
                "사람 목적어 앞 → a", "사람 목적어 앞에 a를 잘 붙였어요.",
            ))
    for tmpl, thing, ko in THINGS_A:
        wrong = "al " + thing[3:] if thing.startswith("el ") else "a " + thing
        items.append(Item(
            "personal_a", tmpl, thing, wrong, ko,
            "인칭의 a는 **사람**이 직접목적어일 때만 붙여요. 사물이나 장소 앞에는 붙이지 않아요. "
            f"{josa(thing, '은/는')} 사람이 아니라서 a 없이 써요. (비교: Veo a mi madre. 사람이라서 a)",
            "사물·장소 목적어 → a 없음", "사물 목적어 앞에 a를 붙이지 않고 잘 썼어요.",
        ))
    return items


# ---------------------------------------------------------------- preterite vs imperfect

DONE = [  # verb, complement, Korean past
    ("ir", "al cine", "영화관에 갔어요"), ("comer", "paella", "빠에야를 먹었어요"), ("ver", "una película", "영화를 봤어요"),
    ("estudiar", "tres horas", "세 시간 동안 공부했어요"), ("escribir", "una carta", "편지를 썼어요"),
    ("leer", "un libro entero", "책 한 권을 다 읽었어요"), ("comprar", "un coche", "차를 샀어요"),
    ("hacer", "la tarea", "숙제를 했어요"), ("llegar", "tarde a clase", "수업에 늦게 도착했어요"),
]
TIMES = [("Ayer", "어제"), ("Anoche", "어젯밤에"), ("El sábado pasado", "지난 토요일에"), ("El año pasado", "작년에")]
DONE_SUBJECTS = [("", "yo", "저는"), ("", "nosotros", "우리는"), ("mi hermana ", "él", "제 여동생은"), ("mis amigos ", "ellos", "제 친구들은")]
HABITS = [("jugar", "al fútbol", "축구를 하곤 했어요"), ("ir", "a la playa", "바다에 가곤 했어요"),
          ("ver", "dibujos animados", "만화 영화를 보곤 했어요"), ("comer", "mucho chocolate", "초콜릿을 많이 먹곤 했어요"),
          ("leer", "cómics", "만화책을 읽곤 했어요")]


def gen_past_tense():
    items = []
    for verb, comp, ko in DONE:
        for t_es, t_ko in TIMES:
            if t_es == "El año pasado" and verb in ("llegar", "hacer", "estudiar"):
                continue  # "last year I studied three hours" reads oddly
            for s_es, p, s_ko in DONE_SUBJECTS:
                right = form(verb, "indicativo", "pretérito-perfecto-simple", p)
                wrong = form(verb, "indicativo", "pretérito-imperfecto", p)
                items.append(Item(
                    "past_tense", f"{t_es} {s_es}{{X}} {comp}.", right, wrong, f"{t_ko} {s_ko} {ko}.",
                    f"'{t_es.lower()}'처럼 **끝난 시점이 분명하고 한 번 일어나고 끝난 일**은 단순과거(pretérito indefinido)를 써요. "
                    f"불완료과거({wrong})는 과거의 습관이나 배경(~하곤 했다, ~하고 있었다)에 써요.",
                    f"끝난 일({t_es.lower()}) → 단순과거 {right}", "끝난 일에 단순과거를 잘 썼어요.",
                ))
    for verb, comp, ko in HABITS:
        for kid in ("niño", "niña"):
            right = form(verb, "indicativo", "pretérito-imperfecto", "yo")
            wrong = form(verb, "indicativo", "pretérito-perfecto-simple", "yo")
            items.append(Item(
                "past_tense", f"Cuando era {kid}, siempre {{X}} {comp}.", right, wrong, f"어렸을 때 저는 항상 {ko}.",
                "'어렸을 때 항상 ~하곤 했다'처럼 **과거의 습관이나 반복**은 불완료과거(pretérito imperfecto)를 써요. "
                f"단순과거({wrong})는 한 번 일어나고 끝난 일에 써요. siempre, cada día, de niño 같은 말이 힌트예요.",
                f"과거의 습관(siempre) → 불완료과거 {right}", "과거의 습관에 불완료과거를 잘 썼어요.",
            ))
    return items


GENERATORS = {
    "article_gender": gen_article_gender, "agreement": gen_agreement, "ser_estar": gen_ser_estar,
    "subjunctive": gen_subjunctive, "gustar": gen_gustar, "si_clause": gen_si_clause, "personal_a": gen_personal_a,
    "past_tense": gen_past_tense,
}

# ---------------------------------------------------------------- rendering

PROMPTS = [
    '이 문장 맞나요? 틀렸으면 고쳐 주세요: "{s}"',
    '"{s}" 이거 문법적으로 맞아요?',
    "첨삭 부탁드려요: {s}",
    "{s}\n이렇게 써도 돼요?",
    '스페인어 문장 확인 좀 해 주세요. "{s}"',
    "이 문장 자연스러워요? {s}",
]
MULTI_PROMPTS = ['제가 쓴 작문 첨삭해 주세요: "{s}"', "스페인어로 일기를 써 봤어요. 틀린 데 고쳐 주세요.\n{s}",
                 '이 글 문법 확인 부탁드려요: "{s}"']
OPENERS = ["", "", "거의 다 맞았어요! 한 군데만 고치면 돼요.\n\n", "좋은 시도예요! 한 부분만 고쳐 볼게요.\n\n"]


def exercise(rng, pool, item):
    other = rng.choice([o for o in pool if o.correct != item.correct] or pool)
    options = [other.right, other.wrong]
    rng.shuffle(options)
    return f"빈칸을 채워 보세요: {other.sentence('___')} ({' / '.join(options)})"


def check_lines(item, used, ok):
    """Check first, verdict after: name the rule, quote what the learner wrote, then judge it.

    A 2B model that opens with the verdict commits before looking at the risky word.
    Both renderings share this opening, so it can't tell right from wrong until it quotes the form.
    """
    return f"**확인할 점:** {item.rule_short}\n**쓴 형태:** '{used}' {'✓' if ok else '✗'}\n\n"


def render_wrong(rng, item, pool):
    return example(
        rng.choice(PROMPTS).format(s=item.incorrect),
        f"{check_lines(item, item.wrong, False)}{rng.choice(OPENERS)}**고친 문장:** {item.correct}\n({item.ko})\n\n"
        f"**무엇이 틀렸나요?**\n'{item.wrong}' → '{item.right}'\n\n"
        f"**왜 그런가요?**\n{item.rule}\n\n"
        f"**연습 문제**\n{exercise(rng, pool, item)}",
        kind="corr_wrong", error_type=item.type,
    )


def render_ok(rng, item):
    return example(
        rng.choice(PROMPTS).format(s=item.correct),
        f"{check_lines(item, item.right, True)}네, 맞아요! 👍 틀린 곳이 없어요. 자연스러운 문장이에요.\n\n"
        f"{item.correct}\n({item.ko})\n\n잘한 점: {item.ok_note}",
        kind="corr_ok", error_type=item.type,
    )


def render_multi(rng, parts):
    """2-3 sentences, 1-3 of them wrong. Only the wrong ones get listed."""
    text = " ".join(it.incorrect if bad else it.correct for it, bad in parts)
    fixed = " ".join(it.correct for it, _ in parts)
    fixes = [f"{i}. '{it.wrong}' → '{it.right}': {it.rule_short}" for i, (it, _) in
             enumerate([p for p in parts if p[1]], 1)]
    return example(
        rng.choice(MULTI_PROMPTS).format(s=text),
        f"**고친 글:**\n{fixed}\n\n**고친 부분 ({len(fixes)}곳):**\n" + "\n".join(fixes) +
        "\n\n나머지 부분은 잘 썼어요! 틀린 부분은 모두 한국어에 없는 문법이라 자주 틀리는 곳이에요.",
        kind="corr_multi", error_types=[it.type for it, bad in parts if bad],
    )


def main():
    rng = random.Random(SEED)
    banned = eval_sentences()
    pools, rows, dropped = {}, [], 0
    for name, gen in GENERATORS.items():
        items = gen()
        clean = [it for it in items if not (is_contaminated(it.incorrect, banned) or is_contaminated(it.correct, banned))]
        dropped += len(items) - len(clean)
        pools[name] = clean
        chosen = rng.sample(clean, min(PER_TYPE[name], len(clean)))
        n_ok = round(len(chosen) * CORRECT_SHARE)
        rows += [render_wrong(rng, it, clean) for it in chosen]
        rows += [render_ok(rng, it) for it in chosen[:n_ok]]  # paired with their wrong twins above
        print(f"  {name:15} {len(clean):4} candidates -> {len(chosen)} wrong + {n_ok} correct-as-is")
    print(f"  dropped {dropped} items that match eval sentences")

    all_items = [it for pool in pools.values() for it in pool]
    for _ in range(N_MULTI):
        parts = rng.sample(all_items, rng.choice([2, 3]))
        flags = [rng.random() < 0.7 for _ in parts]
        if not any(flags):
            flags[0] = True
        rows.append(render_multi(rng, list(zip(parts, flags))))

    rng.shuffle(rows)
    write_jsonl(rows, OUT_DIR / "sft_corrections.jsonl")


if __name__ == "__main__":
    main()
