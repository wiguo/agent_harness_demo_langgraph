"""Retrieval internals that run without a key: chunking and top-k scoring."""

import math

from retrieval.build_index import chunk_markdown
from retrieval.search import cosine, top_k


def test_chunk_markdown_splits_by_h2_with_doc_title():
    md = "# My Policy\n\nintro\n\n## Section A\n\nAlpha text.\n\n## Section B\n\nBeta text."
    chunks = chunk_markdown(md, "fallback")
    assert [c["section"] for c in chunks] == ["Section A", "Section B"]
    assert all(c["doc_title"] == "My Policy" for c in chunks)
    assert chunks[0]["text"] == "Alpha text."


def test_cosine_identity_and_orthogonality():
    assert math.isclose(cosine([1, 2, 3], [1, 2, 3]), 1.0)
    assert math.isclose(cosine([1, 0], [0, 1]), 0.0, abs_tol=1e-9)


def test_top_k_ranks_and_thresholds():
    entries = [
        {"doc_title": "D", "section": "near", "text": "a", "embedding": [1, 0.1]},
        {"doc_title": "D", "section": "far", "text": "b", "embedding": [0.1, 1]},
        {"doc_title": "D", "section": "close", "text": "c", "embedding": [0.9, 0.2]},
    ]
    result = top_k([1, 0], entries, 3, min_score=0.5)
    assert [r["section"] for r in result] == ["near", "close"]
    assert result[0]["score"] > result[1]["score"]
