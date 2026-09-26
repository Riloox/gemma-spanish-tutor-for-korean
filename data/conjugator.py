"""Thin wrapper around verbecc: correct Spanish conjugations + Korean tense names.

Why a wrapper: training data must be *correct by construction*. This module is the
single source of truth for verb forms, so it refuses anything it isn't sure about:
  - verbs verbecc doesn't know (it would *guess* them with an ML model)
  - `vos` forms (verbecc's voseo tables have errors, e.g. imperative "dá")
"""

import json
import logging
from functools import lru_cache

logging.getLogger("verbecc").setLevel(logging.ERROR)
from verbecc import CompleteConjugator  # noqa: E402

PERSONS = ["yo", "tú", "él", "nosotros", "vosotros", "ellos"]
PERSON_KO = {
    "yo": "yo(나)", "tú": "tú(너)", "él": "él(그)",
    "nosotros": "nosotros(우리)", "vosotros": "vosotros(너희, 스페인)", "ellos": "ellos(그들)",
}

# (verbecc mood, verbecc tense) -> Korean name used in prompts.
# Names match the eval set's wording so the model learns one consistent vocabulary.
TENSES = {
    ("indicativo", "presente"): "직설법 현재",
    ("indicativo", "pretérito-perfecto-simple"): "단순과거(pretérito indefinido)",
    ("indicativo", "pretérito-imperfecto"): "불완료과거(pretérito imperfecto)",
    ("indicativo", "futuro"): "단순미래",
    ("indicativo", "pretérito-perfecto-compuesto"): "현재완료(pretérito perfecto compuesto)",
    ("indicativo", "pretérito-pluscuamperfecto"): "과거완료(pretérito pluscuamperfecto)",
    ("condicional", "presente"): "조건법(condicional)",
    ("subjuntivo", "presente"): "접속법 현재",
    ("subjuntivo", "pretérito-imperfecto-1"): "접속법 과거(pretérito imperfecto de subjuntivo)",
    ("subjuntivo", "pretérito-perfecto"): "접속법 현재완료",
    ("imperativo", "afirmativo"): "긍정 명령법",
}

_cg = CompleteConjugator(lang="es")


class UnknownVerb(ValueError):
    pass


@lru_cache(maxsize=None)
def conjugate(infinitive):
    """Return {(mood, tense): {person: form}} for the tenses/persons we teach."""
    data = json.loads(str(_cg.conjugate(infinitive, conjugate_pronouns=False)))
    if data["verb"]["predicted"]:
        raise UnknownVerb(f"{infinitive}: not in verbecc's dictionary, forms would be guessed")
    table = {}
    for (mood, tense) in TENSES:
        rows = data["moods"][mood][tense]
        # For the imperfect subjunctive, the -ra form is the main one; also accept -se below.
        table[(mood, tense)] = {r["pr"]: r["c"][0] for r in rows if r.get("pr") in PERSONS}
    table[("subjuntivo", "pretérito-imperfecto-2")] = {
        r["pr"]: r["c"][0] for r in data["moods"]["subjuntivo"]["pretérito-imperfecto-2"] if r.get("pr") in PERSONS
    }
    table["participio"] = data["moods"]["participo"]["participo"][0]["c"][0]
    table["gerundio"] = data["moods"]["gerundio"]["gerundio"][0]["c"][0]
    return table


def form(infinitive, mood, tense, person):
    return conjugate(infinitive)[(mood, tense)][person]
