"""query_cases: a typed query surface over cases.csv.

Not text-to-SQL — the pydantic schema is an allowlist, so destructive or
malformed queries are unrepresentable rather than rejected. LangGraph's
ToolNode validates arguments against this schema before the function runs;
a validation failure becomes an error ToolMessage the model can recover from.
"""

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

_CSV = Path(__file__).parent.parent / "data" / "cases.csv"

FilterField = Literal["case_id", "date", "category", "channel", "sentiment",
                      "resolution_hours", "region", "status", "month"]
GroupField = Literal["category", "channel", "region", "status", "sentiment", "month"]
Scalar = Union[str, int, float]


@lru_cache(maxsize=1)
def load_cases() -> tuple[dict, ...]:
    """CSV into memory once per process (cached across warm invocations)."""
    rows = []
    with _CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["resolution_hours"] = (
                None if row["resolution_hours"] == "" else float(row["resolution_hours"])
            )
            row["month"] = row["date"][:7]  # derived, e.g. "2026-07"
            rows.append(row)
    return tuple(rows)


class Filter(BaseModel):
    field: FilterField
    op: Literal["eq", "neq", "gt", "gte", "lt", "lte", "in"]
    value: Union[Scalar, list[Scalar]]


class Query(BaseModel):
    metric: Literal["count", "avg", "sum", "min", "max", "list"]
    field: Optional[Literal["resolution_hours"]] = Field(
        default=None, description="Numeric field to aggregate. Required for avg/sum/min/max."
    )
    group_by: Optional[GroupField] = None
    filters: Optional[list[Filter]] = None
    sort: Optional[Literal["value_desc", "value_asc"]] = None
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _check(self):
        if self.metric in ("avg", "sum", "min", "max") and not self.field:
            raise ValueError("field is required for avg/sum/min/max")
        # A malformed month value would silently match nothing and produce a
        # confidently wrong "0". Reject it so the model gets a recoverable error.
        for f in self.filters or []:
            if f.field == "month":
                values = f.value if isinstance(f.value, list) else [f.value]
                if not all(re.fullmatch(r"\d{4}-\d{2}", str(v)) for v in values):
                    raise ValueError(
                        f'month filter values must be "YYYY-MM", e.g. "2026-07" — got {f.value!r}'
                    )
        return self


def _cmp(a, b) -> float:
    """Ordered comparison: numeric when both sides are numeric, otherwise
    lexicographic — which is correct for ISO dates ("2026-07-02" < "2026-08-01")."""
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return (str(a) > str(b)) - (str(a) < str(b))


def _matches(row_value, op: str, value) -> bool:
    if op == "eq":
        return str(row_value) == str(value)
    if op == "neq":
        return str(row_value) != str(value)
    if op == "in":
        return isinstance(value, list) and str(row_value) in [str(v) for v in value]
    # Ordered ops: null/empty (open cases) never passes.
    if row_value is None or row_value == "":
        return False
    c = _cmp(row_value, value)
    return {"gt": c > 0, "gte": c >= 0, "lt": c < 0, "lte": c <= 0}[op]


def _aggregate(rows: list[dict], metric: str, field: str | None):
    if metric == "count":
        return len(rows)
    values = [r[field] for r in rows if r[field] is not None]
    if not values:
        return None
    if metric == "sum":
        return sum(values)
    if metric == "avg":
        return round(sum(values) / len(values), 1)
    return min(values) if metric == "min" else max(values)


def run_query(q: Query) -> dict:
    filters = q.filters or []
    rows = [r for r in load_cases()
            if all(_matches(r[f.field], f.op, f.value) for f in filters)]

    if q.metric == "list":
        out = [{k: v for k, v in r.items() if k != "month"} for r in rows[: q.limit]]
    elif q.group_by:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r[q.group_by], []).append(r)
        out = [{"group": g, "value": _aggregate(members, q.metric, q.field)}
               for g, members in groups.items()]
    else:
        out = [{"value": _aggregate(rows, q.metric, q.field)}]

    if q.sort and q.metric != "list":
        out.sort(key=lambda r: (r["value"] is None, r["value"]),
                 reverse=q.sort == "value_desc")
    out = out[: q.limit]

    return {"rows": out, "row_count": len(out),
            "filters_applied": [f.model_dump() for f in filters]}


def _execute(**kwargs) -> str:
    return json.dumps(run_query(Query(**kwargs)))


query_cases_tool = StructuredTool.from_function(
    func=_execute,
    name="query_cases",
    description=(
        "Aggregate the customer complaint case dataset (24 cases). Columns: case_id, "
        "date (YYYY-MM-DD), category (Billing | Delivery | Product Quality | Service | "
        "Data/Privacy), channel (Email | Phone | Chat | Web Form), sentiment (negative | "
        "neutral | positive), resolution_hours (empty while a case is open), region "
        "(North | South | East | West), status (open | resolved | closed), month "
        '(derived, "YYYY-MM"). Use for anything numeric.'
    ),
    args_schema=Query,
)
