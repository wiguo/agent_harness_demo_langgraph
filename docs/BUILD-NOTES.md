# Build Notes — process, tooling, and what the eval caught

An honest record of how this agent was built, kept because the process is part of the
deliverable: design choices and verification discipline matter more than the artifact.

## Process timeline

1. **Spec first** (`docs/SPEC.md`). The harness requirements — loop, termination, error
   policy, tool design rules, eval strategy — were written and argued before code, so
   every implementation decision had something to be checked against.
2. **The graph, explicitly.** An explicit `StateGraph` (agent node, `ToolNode`,
   conditional edge) rather than the prebuilt agent constructor, so control flow stays
   visible: where the loop terminates, where errors are absorbed, where the budget bites.
3. **Keyless tests before any API spend.** The loop's contract is deterministic, so it's
   pinned with a scripted fake model injected into `build_graph` — dispatch, recoverable
   errors, the step ceiling — plus the query reducer against hand-verified numbers.
   `pytest` needs no key and no network.
4. **Live eval** (`scripts/eval.py`): 8 cases asserting trajectories, exact numbers,
   citations, and forbidden matches (no invented figures).
5. **Fix what the eval caught** (below), re-run to 8/8, deploy.

## Where AI tooling was used

The spec, design decisions, and review were human-owned; **Claude Code** drafted the
implementation against the spec and drove the test/eval cycle. Every line was reviewed,
and the tests and eval are the mechanism that made AI-drafted code trustworthy — they are
also what caught the bug below.

## The bug the eval caught: the confident wrong zero

During development, a date-range filter with a malformed value silently matched no rows,
and the agent answered *"We received 0 complaints in July 2026"* — fluent, cited-looking,
and wrong. No crash, no error event, nothing in the trace flagged failure; only the eval
case asserting the true count (13) caught it.

**Fix:** ordered comparisons handle ISO dates correctly, open cases (null hours) never
pass ordered filters, and a malformed `month` value is now a **validation error carrying
the expected format** — a recoverable ToolMessage the model corrects in one step — never
a silent zero-row match. Regression tests pin all three.

**Lesson:** the worst agent failure class isn't a crash — it's a plausible wrong answer.
Crashes surface themselves; wrong answers need evals. This incident is why the eval
asserts *forbidden* content (no invented numbers) and not just expected content.

## What the framework provides vs. what the harness still owed

LangGraph supplies the loop (`StateGraph` + conditional edge), transcript state
(`add_messages`), tool dispatch with pydantic validation (`ToolNode`), errors-as-data
(`handle_tool_errors`), the step ceiling (`recursion_limit`), and native LangSmith
tracing. On top of that, `graph_agent/runner.py` adds the forced honest summary when the
ceiling is hit. Token/wall-clock budget *enforcement* and duplicate-call interception are
documented gaps (see the README scorecard) — observed in traces, not yet enforced in the
loop. Enumerating exactly where the framework's guarantees end is treated here as part of
using one responsibly.

## Deliberately not built

Multi-agent hierarchies, a vector DB, auth, CI/CD, Docker, a designed UI, a third tool.
`get_case_details(case_id)` is intentionally absent — it's the extension that shows
whether the tool abstraction is actually clean (one new file, one line in
`tools/__init__.py`).
