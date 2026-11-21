# main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import answer, ingest, col


@asynccontextmanager
async def lifespan(app: FastAPI):
    if col.count() == 0:  # indexe au premier démarrage
        ingest()
    yield


app = FastAPI(title="Assistant RH Acme", lifespan=lifespan)


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskOut(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": col.count()}


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    try:
        text, _, metas = answer(body.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur du LLM : {e}")
    return AskOut(answer=text, sources=sorted({m["source"] for m in metas}))