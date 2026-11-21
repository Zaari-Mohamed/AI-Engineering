# rag.py : pipeline RAG complet (retrieve + answer) avec Groq
# le gain vraiment de langgraph est de pouvoir faire des graphes d'etats plus complexes avec plusieurs LLMs et plusieurs chaines, et de pouvoir faire des transitions conditionnelles entre les noeuds du graphe.
# ca se voit quand le workflow est plus complexe, par exemple si on veut faire un workflow avec plusieurs LLMs differents (ex: groq + openai) et plusieurs chaines, et des transitions conditionnelles entre les noeuds du graphe.
import glob, os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)
MODEL = "openai/gpt-oss-20b"
llm = Groq()
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = chromadb.PersistentClient(path="./chroma_db")
col = db.get_or_create_collection("docs", embedding_function=emb)

SYSTEM = (
    "Tu es l'assistant RH d'MEDZA. Réponds uniquement à partir des extraits "
    "fournis dans <contexte>. Cite la source entre crochets, par exemple "
    "[conges.md]. Si la réponse n'y est pas, réponds exactement : "
    "« Je ne trouve pas cette information dans les documents. »"
)


def chunk(text, size=500, overlap=80):
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def ingest(folder="docs", size=500, overlap=80):
    for path in glob.glob(f"{folder}/*.md"):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        name = os.path.basename(path)
        for i, part in enumerate(chunk(text, size, overlap)):
            col.upsert(ids=[f"{name}-{i}"], documents=[part],
                       metadatas=[{"source": name}])
    print("Chunks indexés :", col.count())


def retrieve(question, k=4):
    res = col.query(query_texts=[question], n_results=min(k, col.count()))
    return res["documents"][0], res["metadatas"][0]


def answer(question, k=4):
    docs, metas = retrieve(question, k)
    context = "\n\n".join(f"[{m['source']}] {d}" for d, m in zip(docs, metas))
    prompt = f"<contexte>\n{context}\n</contexte>\n\nQuestion : {question}"
    r = llm.chat.completions.create(
        model=MODEL,
        max_tokens=1500,  # large : gpt-oss "réfléchit" avant de répondre
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return r.choices[0].message.content or "", docs, metas


if __name__ == "__main__":
    ingest()
    if col.count() == 0:
        print("Aucun document indexé : crée un dossier 'docs' avec des fichiers .md")
        raise SystemExit
    while True:
        q = input("\nQuestion > ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        text, _, _ = answer(q)
        print(text)