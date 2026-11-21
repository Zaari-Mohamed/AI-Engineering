# check_env.py : vérifie que tout est prêt
import sys, os, importlib
from dotenv import load_dotenv

load_dotenv()

print("Python :", sys.version.split()[0])
assert sys.version_info >= (3, 10), "Python 3.10 ou plus est requis"

for pkg in ["groq", "pydantic", "numpy", "sentence_transformers",
            "chromadb", "fastapi", "langchain_groq", "langgraph"]:
    try:
        importlib.import_module(pkg)
        print("OK     ", pkg)
    except ImportError:
        print("MANQUANT", pkg)

print("Clé API :", "OK" if os.getenv("GROQ_API_KEY") else "MANQUANTE")