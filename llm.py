"""Model configuration + a raw OpenAI client for embeddings.

The agent itself uses LangChain's ChatOpenAI (see graph_agent/graph.py);
retrieval keeps a plain OpenAI client because embedding one query needs no
framework. A tiny .env loader avoids a dependency; on Vercel the file doesn't
exist and env vars come from the dashboard.
"""

import os
from pathlib import Path

_ROOT = Path(__file__).parent


def load_env(path: Path = _ROOT / ".env") -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env()

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_client = None


def get_client():
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
        from openai import OpenAI

        _client = OpenAI()
    return _client
