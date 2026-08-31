"""★ THE HARNESS, as an explicit LangGraph StateGraph. Read this first.

        START
          │
          ▼
       ┌─────────┐   tool_calls?   ┌─────────┐
       │  agent  │ ──────────────► │  tools  │
       │ (model) │                 │(execute)│
       └─────────┘ ◄────────────── └─────────┘
          │  no tool calls
          ▼
         END  (the answer)

Same loop as a hand-written harness — call model, execute requested tools,
feed results back, repeat — but expressed as a graph. LangGraph supplies the
state merging (add_messages), the tool execution node (ToolNode, which turns
tool exceptions AND argument-validation failures into error ToolMessages the
model can read and recover from), and the routing (tools_condition). What it
does NOT supply — step budgets beyond recursion_limit, duplicate-call
interception, result-size caps, forced summaries — lives in runner.py.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from prompts import SYSTEM_PROMPT
from tools import TOOLS


class State(TypedDict):
    # add_messages appends each node's returned messages to the transcript.
    messages: Annotated[list, add_messages]


def build_graph(model=None, tools=TOOLS):
    """Compile the agent graph. `model` is injected so tests can pass a
    scripted fake; by default it's ChatOpenAI with the tools bound."""
    if model is None:
        from langchain_openai import ChatOpenAI

        from llm import MODEL

        model = ChatOpenAI(model=MODEL, temperature=0).bind_tools(tools)

    def agent(state: State):
        response = model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool_calls -> "tools", else END
    graph.add_edge("tools", "agent")  # results always go back to the model
    return graph.compile()
