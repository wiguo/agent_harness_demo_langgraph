"""Thin FastAPI app: parse -> run_events() -> stream NDJSON.
All agent logic lives in graph_agent/ — this file only adapts HTTP to it.
Vercel's native FastAPI runtime serves this module's `app` directly.
"""

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from graph_agent.runner import run_events

app = FastAPI()

_PAGE = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/")
def home() -> HTMLResponse:
    return HTMLResponse(_PAGE)


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        return JSONResponse({"error": "messages[] required"}, status_code=400)
    # Stateless server: the client posts its history each turn; we sanitise and cap it.
    messages = [
        {"role": m["role"], "content": str(m.get("content", ""))}
        for m in raw
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ][-20:]

    def events():
        try:
            for event in run_events(messages):
                yield json.dumps(event) + "\n"
        except Exception as err:
            print(f"agent error: {err!r}", file=sys.stderr)
            yield json.dumps({"type": "error",
                              "message": "The agent hit an unrecoverable error. Try again."}) + "\n"
        finally:
            _flush_traces()

    return StreamingResponse(events(), media_type="application/x-ndjson")


def _flush_traces() -> None:
    """Serverless freezes the process after the response; block briefly so
    LangSmith's background uploader finishes, or runs show as 'pending'."""
    if os.environ.get("LANGSMITH_TRACING") == "true":
        try:
            from langchain_core.tracers.langchain import wait_for_all_tracers

            wait_for_all_tracers()
        except Exception as err:
            print(f"trace flush failed: {err!r}", file=sys.stderr)
