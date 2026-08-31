"""Stream the graph as harness events — one NDJSON event per step, tool call,
and final answer, consumed identically by the web UI and the eval script.

Also the home of the guardrails LangGraph doesn't give you out of the box:
the step ceiling (recursion_limit) with a forced no-tools summary when it's
hit, instead of an exception surfacing to the user.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from prompts import SYSTEM_PROMPT

MAX_STEPS = 6  # model calls per run
# One graph step per node execution: N agent calls interleave N-1 tool nodes.
RECURSION_LIMIT = 2 * MAX_STEPS

_default_graph = None


def _get_default_graph():
    global _default_graph
    if _default_graph is None:
        from .graph import build_graph

        _default_graph = build_graph()
    return _default_graph


def _summary_model():
    from langchain_openai import ChatOpenAI

    from llm import MODEL

    return ChatOpenAI(model=MODEL, temperature=0)  # tools NOT bound: must answer


def run_events(messages, graph=None, summary_model=None):
    """Yield step / tool / final events for one user turn.
    `messages` is openai-style [{"role", "content"}, ...]; add_messages coerces."""
    graph = graph or _get_default_graph()
    transcript = list(messages)
    step = 0
    args_by_id = {}
    final_event = None

    try:
        stream = graph.stream(
            {"messages": messages},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="updates",
        )
        # Iterate to natural exhaustion — returning/breaking mid-stream would
        # force-close the generator and LangGraph's trace would record a
        # spurious GeneratorExit error. After the final answer the graph is at
        # END and the stream yields nothing more anyway.
        for update in stream:
            for node, out in update.items():
                for msg in out["messages"]:
                    transcript.append(msg)
                    if node == "agent":
                        step += 1
                        usage = getattr(msg, "usage_metadata", None) or {}
                        yield {"type": "step", "step": step,
                               "usage": {"total_tokens": usage.get("total_tokens")}}
                        for tc in msg.tool_calls or []:
                            args_by_id[tc["id"]] = tc["args"]
                        if not msg.tool_calls:
                            final_event = {"type": "final", "content": msg.content,
                                           "stop_reason": "answer"}
                    else:  # tools node -> ToolMessage(s)
                        ok = getattr(msg, "status", None) != "error"
                        yield {"type": "tool", "name": msg.name, "ok": ok,
                               "args": args_by_id.get(msg.tool_call_id, {}),
                               "preview": _preview(msg.content, ok)}
        if final_event is not None:
            yield final_event
            return
    except GraphRecursionError:
        # Step ceiling hit mid-task: one forced no-tools call to summarise
        # honestly instead of leaking an exception to the user.
        model = summary_model or _summary_model()
        res = model.invoke([
            SystemMessage(SYSTEM_PROMPT),
            *transcript,
            HumanMessage(
                "The run hit its step budget. Without calling any more tools, "
                "summarise what you found so far and answer as best you can. "
                "Say explicitly what is still unknown."
            ),
        ])
        yield {"type": "final", "content": res.content,
               "stop_reason": "max_steps_exceeded"}


def _preview(content, ok: bool) -> str:
    """Short human-readable summary of a tool result, for the UI event stream."""
    text = content if isinstance(content, str) else json.dumps(content)
    if not ok:
        return text[:300]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text[:300]
    if "chunks" in data:
        if not data["chunks"]:
            return "no chunks above threshold"
        return " · ".join(
            f"{c['doc_title']} §{c['section']} ({c['score']:.2f})" for c in data["chunks"]
        )
    if "rows" in data:
        return f"{data['row_count']} row(s): {json.dumps(data['rows'])[:300]}"
    return text[:300]
