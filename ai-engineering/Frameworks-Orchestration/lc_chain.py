# lc_chain.py : partie A (LCEL) juste exemple simple pour voir le mecanisme mais generalement ca sutilise quand ya plus de chaine et plusieurs LLMs et aussi des LLMs differents (ex: groq + openai) et des chaines plus complexes.
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=300)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu résumes le texte fourni en {n} phrases, en français."),
    ("human", "{texte}"),
])
chain = prompt | llm | StrOutputParser()  # le pipe = enchaînement

print(chain.invoke({"n": 2, "texte": "Le RAG combine une recherche de documents "
    "pertinents et un modèle de langage qui génère la réponse à partir d'eux."}))