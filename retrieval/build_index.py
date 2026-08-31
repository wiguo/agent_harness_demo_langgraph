"""python -m retrieval.build_index — chunk the markdown policies by H2 section,
embed each chunk once, write data/index.json (committed, so runtime never pays
for it; Vercel's filesystem is read-only anyway).
"""

import json
import re
from pathlib import Path

_ROOT = Path(__file__).parent.parent
DOCS_DIR = _ROOT / "data" / "documents"
OUT_FILE = _ROOT / "data" / "index.json"
MAX_CHUNK_CHARS = 2400  # ~600 tokens; sections here never exceed it


def chunk_markdown(markdown: str, fallback_title: str) -> list[dict]:
    title_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else fallback_title
    chunks = []
    sections = re.split(r"^##\s+", markdown, flags=re.MULTILINE)[1:]  # drop preamble
    for section in sections:
        heading, _, body = section.partition("\n")
        text = body.strip()
        if not text:
            continue
        # Oversized sections split on paragraph boundaries.
        buffer = ""
        for para in re.split(r"\n\n+", text):
            if len(buffer) + len(para) > MAX_CHUNK_CHARS and buffer.strip():
                chunks.append({"doc_title": doc_title, "section": heading.strip(),
                               "text": buffer.strip()})
                buffer = ""
            buffer += para + "\n\n"
        if buffer.strip():
            chunks.append({"doc_title": doc_title, "section": heading.strip(),
                           "text": buffer.strip()})
    return chunks


def main() -> None:
    from llm import EMBEDDING_MODEL, get_client

    files = sorted(DOCS_DIR.glob("*.md"))
    chunks = [c for f in files for c in chunk_markdown(f.read_text(encoding="utf-8"), f.name)]
    print(f"{len(files)} documents -> {len(chunks)} chunks; embedding...")

    res = get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=[f"{c['doc_title']} — {c['section']}\n{c['text']}" for c in chunks],
    )
    index = [{**c, "embedding": res.data[i].embedding} for i, c in enumerate(chunks)]

    OUT_FILE.write_text(json.dumps(index), encoding="utf-8")
    print(f"Wrote {OUT_FILE} ({OUT_FILE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
