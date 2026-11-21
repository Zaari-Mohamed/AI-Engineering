# semantic_search.py
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
docs = [
    "Le vétérinaire a vacciné mon chat hier.",
    "Les croquettes pour chiot sont en promotion.",
    "Mon animal refuse de manger depuis deux jours.",
    "Le taux du prêt immobilier a baissé ce mois-ci.",
    "La bourse a clôturé en hausse après l'annonce.",
    "Comment ouvrir un livret d'épargne en ligne ?",
    "Préchauffer le four à 180 degrés pour le gâteau.",
    "Cette recette de tarte utilise de la farine complète.",
    "Mélanger les oeufs et le sucre jusqu'à blanchiment.",
]
doc_vecs = model.encode(docs, normalize_embeddings=True)   # forme (9, 384)
def search(query, k=3):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = doc_vecs @ q               # produit scalaire = cosinus ici
    print(f"\nRequête : {query}")
    for i in np.argsort(-scores)[:k]:
        print(f"  {scores[i]:.2f}  {docs[i]}")
        
for q in ["mon chien est malade", "comment placer mon argent",
          "que cuisiner ce soir"]:
    search(q)
# Pour la recherche par mot clé  : 
# ce qui il faut retenir : la recherche sematique peut trouver des documents pertinents même si les mots exacts de la requête ne sont pas présents dans le document.
# mais la recherche par mot clé ne trouvera que les documents contenant les mots exacts de la requête, ce qui peut être limitatif.
# def keyword_search(query):
#     mots = query.lower().split()
#     print(f"\nRequête (mots-clés) : {query}")
#     trouve = False
#     for doc in docs:
#         communs = [m for m in mots if m in doc.lower()]
#         if communs:
#             trouve = True
#             print(f"  {communs} -> {doc}")
#     if not trouve:
#         print("  Aucun résultat")

# keyword_search("mon chien est malade")   # ligne d'appel commentée aussi
# search("mon chien est malade")