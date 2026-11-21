# Partie B : Chroma avec filtre sur les métadonnées
import chromadb
from chromadb.utils import embedding_functions

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

emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_demo")
col = client.get_or_create_collection("phrases", embedding_function=emb)

themes = ["animaux"] * 3 + ["finance"] * 3 + ["cuisine"] * 3
col.upsert(ids=[f"d{i}" for i in range(len(docs))], documents=docs,
           metadatas=[{"theme": t} for t in themes])

res = col.query(query_texts=["je veux investir"], n_results=2,
                where={"theme": "finance"})

# Affichage lisible : distance (plus petite = plus proche) + phrase
print('\nRequête : "je veux investir" (thème : finance)')
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"  {dist:.2f}  {doc}")