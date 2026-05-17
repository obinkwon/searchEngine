from fastapi import FastAPI
from search_engine import search

app = FastAPI()


@app.get("/search")
def search_api(q: str):
    results = search(q)

    return {"query": q, "results": results}
