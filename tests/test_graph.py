"""Graph/harness contract tests. All keyless: the model is a scripted fake
injected through build_graph(model=...); ToolNode executes the real
query_cases tool (pure CSV work, no API)."""

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from graph_agent.graph import build_graph
from graph_agent.runner import run_events


class FakeModel:
    """Stands in for ChatOpenAI.bind_tools(...): returns scripted AIMessages."""

    def __init__(self, script):
        self._script = list(script)

    def invoke(self, _messages):
        return self._script.pop(0) if len(self._script) > 1 else self._script[0]


def tool_call_msg(name, args, call_id="tc1"):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}])


USER = [{"role": "user", "content": "hi"}]


def collect(script, tools=None, **kwargs):
    graph = (build_graph(model=FakeModel(script), tools=tools)
             if tools else build_graph(model=FakeModel(script)))
    return list(run_events(USER, graph=graph, **kwargs))


def test_tool_round_trip_then_final_answer():
    events = collect([
        tool_call_msg("query_cases", {
            "metric": "count",
            "filters": [{"field": "status", "op": "eq", "value": "open"}],
        }),
        AIMessage(content="There are 4 open cases."),
    ])
    assert [e["type"] for e in events] == ["step", "tool", "step", "final"]
    tool_event = events[1]
    assert tool_event["ok"] is True
    assert '"value": 4' in tool_event["preview"].replace('": ', '": ')
    assert events[-1]["content"] == "There are 4 open cases."
    assert events[-1]["stop_reason"] == "answer"


def test_invalid_args_become_recoverable_error_message():
    events = collect([
        tool_call_msg("query_cases", {"metric": "avg", "field": "refund_amount"}),
        AIMessage(content="recovered"),
    ])
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["ok"] is False
    assert events[-1]["content"] == "recovered"  # model saw the error and continued


def test_tool_exception_becomes_recoverable_error_message():
    class NoArgs(BaseModel):
        pass

    def _boom() -> str:
        raise RuntimeError("kaboom")

    bomb = StructuredTool.from_function(func=_boom, name="bomb",
                                        description="always throws", args_schema=NoArgs)
    events = collect(
        [tool_call_msg("bomb", {}), AIMessage(content="recovered")], tools=[bomb])
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["ok"] is False
    assert "kaboom" in tool_event["preview"]
    assert events[-1]["content"] == "recovered"


def test_step_ceiling_forces_no_tools_summary():
    class LoopingModel:
        def __init__(self):
            self.n = 0

        def invoke(self, _messages):
            self.n += 1
            return tool_call_msg(
                "query_cases", {"metric": "count", "limit": self.n}, f"tc{self.n}")

    class SummaryModel:
        def invoke(self, _messages):
            return AIMessage(content="partial summary")

    graph = build_graph(model=LoopingModel())
    events = list(run_events(USER, graph=graph, summary_model=SummaryModel()))
    final = events[-1]
    assert final["type"] == "final"
    assert final["stop_reason"] == "max_steps_exceeded"
    assert final["content"] == "partial summary"
    assert sum(e["type"] == "step" for e in events) >= 5  # ceiling reached, not one-shot
