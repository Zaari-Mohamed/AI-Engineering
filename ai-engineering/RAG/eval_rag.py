# eval_rag.py
import json
import re
from rag import ingest, retrieve, answer, llm, MODEL

DATASET = [
    {"q": "Combien de jours de congés par an ?", "src": "conges.md", "kw": ["25"]},
    {"q": "Quel est le report maximal de congés ?", "src": "conges.md", "kw": ["5"]},
    {"q": "Quel jour la présence au bureau est-elle obligatoire ?",
     "src": "teletravail.md", "kw": ["mardi"]},
    {"q": "Combien de jours de télétravail par semaine ?",
     "src": "teletravail.md", "kw": ["3"]},
    {"q": "Plafond de remboursement d'un repas professionnel ?",
     "src": "frais.md", "kw": ["25"]},
    {"q": "Sous quel délai soumettre une note de frais ?",
     "src": "frais.md", "kw": ["30"]},
    {"q": "Quelle est la capitale du Japon ?", "src": None, "kw": ["ne trouve pas"]},
]

JUGE = (
    "Tu es un évaluateur strict. On te donne un CONTEXTE et une RÉPONSE. "
    "Dis si toutes les affirmations de la réponse sont soutenues par le contexte. "
    "Une réponse qui dit qu'elle ne trouve pas l'information est fidèle. "
    'Réponds uniquement en JSON, avec "fidele" en premier et un raisonnement '
    'd\'UNE phrase courte : {"fidele": true|false, "raisonnement": "..."}'
)

def juge_fidelite(context, reponse):
    msg = f"<contexte>{context}</contexte>\n<reponse>{reponse}</reponse>"
    try:
        r = llm.chat.completions.create(
            model=MODEL,
            max_tokens=1000,          # marge large pour éviter la coupure
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUGE},
                {"role": "user", "content": msg},
            ],
        )
        raw = r.choices[0].message.content.strip()
        return bool(json.loads(raw)["fidele"])
    except Exception as e:
        print("   (juge en échec :", str(e)[:80], ")")
        return False

def run():
    hit = kw = fid = n_src = 0
    for case in DATASET:
        docs, metas = retrieve(case["q"])
        text, _, _ = answer(case["q"])
        if case["src"]:
            n_src += 1
            hit += case["src"] in {m["source"] for m in metas}
        good_kw = all(w.lower() in text.lower() for w in case["kw"])
        kw += good_kw
        fid += juge_fidelite("\n".join(docs), text)
        print(("OK " if good_kw else "KO "), case["q"], "->", text[:70])
    n = len(DATASET)
    print(f"\nHit rate      : {hit}/{n_src}")
    print(f"Mots-clés     : {kw}/{n}")
    print(f"Fidélité juge : {fid}/{n}")

if __name__ == "__main__":
    ingest()
    run()