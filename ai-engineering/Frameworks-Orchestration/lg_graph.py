# lg_graph.py : partie B (LangGraph)
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from rag import retrieve

load_dotenv(override=True)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=500)

class State(TypedDict, total=False):
    question: str
    route: str
    context: str
    answer: str

def router(state: State):
    q = state["question"].lower()
    metier = any(w in q for w in ["congé", "télétravail", "frais", "remboursement"])
    return {"route": "rag" if metier else "chat"}

def retrieve_node(state: State):
    docs, _ = retrieve(state["question"])
    return {"context": "\n".join(docs)}

def generate_node(state: State):
    ctx = state.get("context", "")
    msg = (f"Contexte :\n{ctx}\n\nQuestion : {state['question']}"
           if ctx else state["question"])
    return {"answer": llm.invoke(msg).content}

g = StateGraph(State)
g.add_node("router", router)
g.add_node("retrieve", retrieve_node)
g.add_node("generate", generate_node)
g.add_edge(START, "router")
g.add_conditional_edges("router", lambda s: s["route"],
                        {"rag": "retrieve", "chat": "generate"})
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
app = g.compile()

for q in ["Combien de jours de congés ai-je ?", "Explique-moi ce qu'est un token."]:
    print(q, "->", app.invoke({"question": q})["answer"], "\n")