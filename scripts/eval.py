"""python scripts/eval.py — 8 scripted cases through the real graph + model.

No framework: a list of cases, each asserting on which tools ran and what the
answer must (or must not) contain. Requires OPENAI_API_KEY (loaded from .env).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_agent.runner import run_events

CASES = [
    {
        "q": "What is the complaint escalation process?",
        "tools": ["search_documents"],
        "must": [r"48\s*hours?", r"complaint handling policy"],
    },
    {
        "q": "How long are complaint records retained?",
        "tools": ["search_documents"],
        "must": [r"24\s*months", r"data privacy policy"],
    },
    {
        "q": "How many complaints did we receive in July 2026?",
        "tools": ["query_cases"],
        "must": [r"\b13\b"],
    },
    {
        "q": "What is the most common complaint category?",
        "tools": ["query_cases"],
        "must": [r"billing", r"\b7\b"],
    },
    {
        "q": "What is the average resolution time by channel?",
        "tools": ["query_cases"],
        "must": [r"42", r"31\.7", r"26\.4", r"18\.4"],
    },
    {
        "q": "What are the top complaint categories, and what do our guidelines "
             "say about handling them?",
        "tools": ["query_cases", "search_documents"],  # the cross-source centrepiece
        "must": [r"billing"],
    },
    {
        # Two honest paths, both pass: attempt query_cases with refund_amount,
        # get a validation error, recover — or read the column list and decline
        # without calling. What must never happen is an invented number.
        "q": "What's our average refund amount?",
        "tools": [],
        "optional": ["query_cases", "search_documents"],
        "must_not": [r"£\s?\d", r"\$\s?\d", r"\b\d+\.\d{2}\b"],
    },
    {
        "q": "Who is the CEO?",
        "tools": [],
        "optional": ["search_documents"],
        "must_not": [r"is the CEO of"],
    },
]


def evaluate(case: dict) -> dict:
    used, answer = [], ""
    for e in run_events([{"role": "user", "content": case["q"]}]):
        if e["type"] == "tool":
            used.append(e["name"])
        if e["type"] == "final":
            answer = e["content"] or ""

    expected = case["tools"]
    optional = case.get("optional", [])
    problems = [f"missing tool {t}" for t in expected if t not in used]
    problems += [f"unexpected tool {t}" for t in used
                 if t not in expected and t not in optional]
    problems += [f"no match /{p}/" for p in case.get("must", [])
                 if not re.search(p, answer, re.IGNORECASE)]
    problems += [f"forbidden match /{p}/" for p in case.get("must_not", [])
                 if re.search(p, answer, re.IGNORECASE)]
    return {"used": used, "answer": answer, "problems": problems}


def main() -> None:
    failures = 0
    for i, case in enumerate(CASES, 1):
        r = evaluate(case)
        ok = not r["problems"]
        failures += not ok
        print(f"\n[{i}/8] {'PASS' if ok else 'FAIL'} — {case['q']}")
        print(f"  tools: {', '.join(r['used']) or '(none)'}")
        if not ok:
            print(f"  problems: {'; '.join(r['problems'])}")
            print(f"  answer: {r['answer'][:400]}")
    print(f"\n{8 - failures}/8 passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
