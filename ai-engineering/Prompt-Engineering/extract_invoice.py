# extract_invoice_fewshot.py (version Groq + few-shot)

# ici few shot veut dire donner au model des exemples pour qui il s'inspire pour donner un resultat précis .
# si on veut wero shot (c'est a dire pas de exemple le resultat peut etre moisn precis) il suffit de ne pas mettre d'exemple dans la variable FEW_SHOT

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from groq import Groq

load_dotenv(override=True)
client = Groq()
MODEL = "openai/gpt-oss-20b"


class Invoice(BaseModel):
    vendor: str
    date: str  # format AAAA-MM-JJ
    total_eur: float
    items: list[str]


SYSTEM = (
    "Tu extrais des données de factures. Réponds UNIQUEMENT avec un objet "
    "JSON valide, sans aucun texte autour. Clés : vendor, date (AAAA-MM-JJ), "
    "total_eur (nombre), items (liste de chaînes)."
)

# --- Exemples few-shot : paires user / assistant (différents des CASES de test) ---
FEW_SHOT = [
    {"role": "user", "content": "<facture>\nPizzeria Roma, 5 mai 2025. 1 pizza 11,00 EUR, "
                               "1 soda 3,00 EUR. Total 14,00 EUR\n</facture>\n\nExtrais les informations."},
    {"role": "assistant", "content": '{"vendor": "Pizzeria Roma", "date": "2025-05-05", '
                                     '"total_eur": 14.0, "items": ["1 pizza", "1 soda"]}'},
    {"role": "user", "content": "<facture>\nFACTURE n°12 - Garage Leroy - 03/02/2025 - "
                                "Pneus 120,00 EUR - TOTAL TTC : 120,00 EUR\n</facture>\n\nExtrais les informations."},
    {"role": "assistant", "content": '{"vendor": "Garage Leroy", "date": "2025-02-03", '
                                     '"total_eur": 120.0, "items": ["Pneus"]}'},
    {"role": "user", "content": "<facture>\nPharmacie Centrale 2025-06-18 : 1 sirop (6,50 EUR), "
                                "1 crème (9,90 EUR) = 16,40 euros\n</facture>\n\nExtrais les informations."},
    {"role": "assistant", "content": '{"vendor": "Pharmacie Centrale", "date": "2025-06-18", '
                                     '"total_eur": 16.4, "items": ["1 sirop", "1 crème"]}'},
]


def clean(raw):
    """Retire les ```json ... ``` si le modèle en met malgré la consigne."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return raw


def extract(text, retries=2):
    prompt = f"<facture>\n{text}\n</facture>\n\nExtrais les informations."
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1500,  # large : gpt-oss "réfléchit" avant de répondre
            temperature=0,
            messages=(
                [{"role": "system", "content": SYSTEM}]
                + FEW_SHOT
                + [{"role": "user", "content": prompt}]
            ),
        )
        raw = clean(resp.choices[0].message.content or "")
        try:
            return Invoice.model_validate_json(raw)
        except ValidationError as e:
            print(f"  tentative {attempt + 1} invalide, nouvelle demande")
            prompt += f"\nTa réponse précédente était invalide ({e}). Corrige."
    raise ValueError("Extraction impossible")


CASES = [
    ("Boulangerie Dupont, 3 mars 2025. 2 baguettes 2,40 EUR, 1 tarte 12,50 EUR. "
     "Total 14,90 EUR", 14.90),
    ("FACTURE n°88 - Garage Martin - 12/01/2025 - Vidange 89,00 EUR - "
     "Filtre 24,50 EUR - TOTAL TTC : 113,50 EUR", 113.50),
    ("Café Lumière 2025-02-14 : 2 cafés (7,00 EUR), 1 jus (4,20 EUR) = 11,20 EUR", 11.20),
    ("Librairie du Port, 20 avril 2025, 3 livres, total à payer 47,60 euros", 47.60),
]

if __name__ == "__main__":
    ok = 0
    for text, expected in CASES:
        inv = extract(text)
        good = abs(inv.total_eur - expected) < 0.01
        ok += good
        print("OK " if good else "KO ", inv.model_dump())
    print(f"\nScore : {ok}/{len(CASES)}")