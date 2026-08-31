"""Cosine top-k over the committed embedding index.

The index loads once per process; each search embeds the query (one API call)
and scores in memory. 19 chunks x 1536 dims — pure Python is plenty; numpy
would be a dependency, not a speedup anyone can feel.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

from llm import EMBEDDING_MODEL, get_client

_INDEX = Path(__file__).parent.parent / "data" / "index.json"

MIN_SCORE = 0.18  # below this, chunks are noise, not evidence
# (text-embedding-3-small cosine scores run low; relevant sections often land
# in the 0.2–0.4 band, so recall beats precision at this corpus size)


@lru_cache(maxsize=1)
def load_index() -> tuple[dict, ...]:
    return tuple(json.loads(_INDEX.read_text(encoding="utf-8")))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def top_k(query_embedding, entries, k: int, min_score: float = MIN_SCORE) -> list[dict]:
    scored = [
        {
            "doc_title": e["doc_title"],
            "section": e["section"],
            "text": e["text"],
            "score": cosine(query_embedding, e["embedding"]),
        }
        for e in entries
    ]
    scored = [c for c in scored if c["score"] >= min_score]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:k]


def search(query: str, k: int) -> list[dict]:
    res = get_client().embeddings.create(model=EMBEDDING_MODEL, input=query)
    return top_k(res.data[0].embedding, load_index(), k)
