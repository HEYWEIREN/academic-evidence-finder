from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from .evaluate import evaluate_all
from .search_engine import ENGINE, SearchConfig


app = FastAPI(title="Academic Evidence Finder")


@app.get("/api/search")
def search(
    q: str = "",
    year: int | None = None,
    topic: str | None = None,
    mode: str = Query("hybrid", pattern="^(hybrid|bm25|semantic)$"),
):
    return ENGINE.search(q, SearchConfig(mode=mode, year=year, topic=topic))


@app.get("/api/papers/{paper_id}")
def paper_detail(paper_id: str):
    paper = ENGINE.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return paper


@app.get("/api/topics")
def topics():
    return {"topics": ENGINE.topics(), "years": ENGINE.years(), "paper_count": ENGINE.paper_count()}


@app.get("/api/evaluate")
def evaluate():
    return evaluate_all()


app.mount("/", StaticFiles(directory="web", html=True), name="web")
