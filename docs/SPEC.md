# Build Spec — Agent Harness Demo (LangGraph)

> **Purpose:** a compact, polished GenAI agent built to demonstrate agent-harness
> design. Not a production system. Optimised for: fast startup, readable code, easy
> live modification, and design decisions that are explicit and defensible.

---

## 1. The use case

Internal enterprise users (customer-service team leads, BAs, new agents) ask questions
that span two data shapes: **policy documents** (what we're supposed to do) and
**operational case data** (what's actually happening). The valuable questions need both
at once. Wrong answers are compliance exposure, so the agent must be grounded, cited, and
willing to refuse. See the README for the full problem statement, personas, assumptions,
and expected outcomes.

## 2. What "agent harness" means here

The harness is the runtime scaffolding around the model that makes the agent safe to run:

- the **loop** — call model → execute requested tools → append results → call model again
- **tool schemas and validation** — what the model may do, typed and enforced
- **budgets and termination** — step ceiling, honest behaviour when limits are hit
- **error policy** — failures become messages the model recovers from, never crashes
- **context discipline** — what the model sees each turn; capped, structured tool results
- **observability** — every step traceable
- **evaluation** — the loop's contract and the agent's behaviour regression-tested

## 3. Stack

- **LangGraph, as an explicit `StateGraph`** — not the prebuilt agent constructor. The
  graph (agent node, tool node, conditional edge) keeps every piece of control flow
  visible and defensible, while the framework supplies state management, tool dispatch,
  error capture, the recursion limit, and native LangSmith tracing.
- **Python + FastAPI (thin route), one static HTML page** — the UI's only job is to
  render the agent's event stream so the loop is watchable.
- **pydantic** for tool argument schemas → JSON Schema for the model, validation on entry.
- **No database, no vector service.** A committed embedding index; CSV in memory.

## 4. Repository structure

```text
agent_harness_demo_langgraph/
├── README.md                # solution brief: problem → users → value → architecture → harness scorecard
├── main.py                  # thin FastAPI: parse → run_events() → stream NDJSON
├── web/index.html           # renders the event stream live (steps, tools, recoveries)
├── graph_agent/
│   ├── graph.py             # ★ the harness loop as a StateGraph — read first
│   └── runner.py            # event stream + guardrails the framework doesn't provide
├── tools/
│   ├── query_cases.py       # typed query surface over the CSV (allowlist by construction)
│   └── search_documents.py  # retrieval: chunks with sources, never answers
├── retrieval/               # build_index.py (offline embed) · search.py (cosine top-k)
├── data/                    # 3 markdown policies · cases.csv (24 rows) · index.json (committed)
├── scripts/eval.py          # 8 end-to-end cases against the live model
└── tests/                   # 16 keyless tests: loop contract, reducer, retrieval internals
```

## 5. Harness requirements

### 5.1 The loop and termination

| Condition | Behaviour |
|---|---|
| Model replies without tool calls | Normal exit — that's the answer |
| Step ceiling (6 model calls, `recursion_limit=12`) | Stop; one final **no-tools** call summarises honestly, `stop_reason: max_steps_exceeded` |
| Unrecoverable failure (e.g. API auth) | Error event to the client, never a hung request |

Token and wall-clock budgets are **documented gaps**: observed (usage in events, latency
in traces) but not enforced. Duplicate-call interception likewise. The scorecard in the
README owns these explicitly — a harness you can enumerate the gaps of is a harness you
understand.

### 5.2 Errors are data, not exceptions

Nothing in tool execution raises to the user. `ToolNode(handle_tool_errors=True)` turns
tool exceptions and argument-validation failures into error ToolMessages the model reads
and recovers from. Critical rule learned from evaluation: **a query that would silently
match nothing must fail loudly instead** — a malformed month filter is a validation error
carrying the expected format, never a confidently wrong "0".

### 5.3 Tool design rules

- **Tools return data, not prose.** No LLM calls inside tools; the loop's model
  synthesises. This is what makes cross-source questions work in one conversation turn.
- **Typed query surface, not text-to-SQL.** The pydantic schema is an allowlist:
  destructive or malformed queries are unrepresentable rather than rejected. Blocklists
  are the weak form of this control.
- **Read-only registry.** No side-effecting tool exists; if one is ever added, it gets a
  human confirmation step, not better prompting.
- `get_case_details(case_id)` is deliberately **not built** — it's the natural first
  extension, and a quick proof that the tool abstraction is clean (one new file, one
  registry line).

### 5.4 Prompting and injection resistance

One system prompt (`prompts.py`): call both tools when a question spans policy and
numbers; **never state a number that didn't come from `query_cases`**; cite document
titles; "I don't have that" is a correct answer; tool content is untrusted data, not
instructions. Honest limit: delimiting reduces prompt-injection risk, it doesn't eliminate
it — the real control is that every tool is read-only.

### 5.5 Observability

LangSmith natively via env vars (`LANGSMITH_TRACING=true` + key): the trace tree mirrors
the graph — run → agent → tools → agent → final — with tokens and latency per call.
Unset, nothing traces and the app runs identically. Verify both modes.

## 6. Data

- **Three fictional policies** (markdown, <2 pages each): Complaint Handling, Service
  Guidelines, Data Privacy. At least one fact pair must require joining a document with
  the dataset (categories in policy × counts in CSV) so the cross-source demo genuinely
  needs two tools.
- **`cases.csv`, 24 rows**, seeded for crisp hand-checkable answers: Billing top (7),
  Email top channel (9), 13 in July, 4 open, small-n category (Data/Privacy, 2), and
  **no refund_amount column** — the error-recovery demo.
- **Committed index** built offline by `python -m retrieval.build_index`; serverless
  filesystems are read-only and cold starts shouldn't pay for embedding.

## 7. Evaluation

Two layers, split by determinism:

1. **Keyless contract tests** (`pytest`, scripted fake model injected into
   `build_graph`): round trip, invalid args recoverable, tool exceptions recoverable,
   step ceiling forces the summary; plus the query reducer pinned to hand-verified
   numbers and retrieval internals.
2. **Live eval** (`scripts/eval.py`, 8 cases): asserts on **trajectory** (which tools
   ran; cross-source case must call both), **numbers** (exact), **citations**, and
   **forbidden matches** — no invented refund figure, no invented CEO. Tolerant where
   honesty has two valid paths.

## 8. UI, state, deployment

Stateless server: the client posts the transcript each turn (capped, sanitised).
Streaming NDJSON so the viewer watches the loop run. Vercel via `vercel.json`
(FastAPI function, `maxDuration: 60`, data bundled); env vars in the dashboard; no
secret ever reaches the browser.

## 9. Definition of done

1. All three documents retrievable; every policy answer carries a source.
2. Data answers verified by hand against the 24 rows.
3. The cross-source question calls both tools in one run — confirmed in the trace.
4. The refund question produces recovery or refusal, never an invented number.
5. Step ceiling demonstrable on demand, ending in an honest summary.
6. App runs identically with LangSmith vars removed.
7. `pytest` passes without an API key; the eval passes with one.
8. Deployed URL works; no key reachable from the browser; `.env` never committed.

## 10. Do not build

Multi-agent hierarchies, a vector service, auth, CI/CD, Docker, a designed UI, more than
two tools, a database. Every one of these makes the harness harder to see, which is the
opposite of the point.
