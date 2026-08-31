"""Reducer tests against the real committed CSV — the same numbers verified by
hand for the demo questions."""

import pytest
from pydantic import ValidationError

from tools.query_cases import Query, run_query


def q(**kwargs):
    return run_query(Query(**kwargs))


def test_count_july_is_13():
    res = q(metric="count", filters=[{"field": "month", "op": "eq", "value": "2026-07"}])
    assert res["rows"][0]["value"] == 13


def test_group_by_category_billing_top_with_7():
    res = q(metric="count", group_by="category", sort="value_desc")
    assert res["rows"][0] == {"group": "Billing", "value": 7}
    assert sum(r["value"] for r in res["rows"]) == 24


def test_avg_resolution_by_channel_skips_open_cases():
    res = q(metric="avg", field="resolution_hours", group_by="channel", sort="value_desc")
    by_channel = {r["group"]: r["value"] for r in res["rows"]}
    assert by_channel == {"Web Form": 42, "Email": 31.7, "Phone": 26.4, "Chat": 18.4}


def test_open_cases_is_4():
    res = q(metric="count", filters=[{"field": "status", "op": "eq", "value": "open"}])
    assert res["rows"][0]["value"] == 4


def test_date_range_filter_works_on_iso_dates():
    res = q(metric="count", filters=[
        {"field": "date", "op": "gte", "value": "2026-07-01"},
        {"field": "date", "op": "lte", "value": "2026-07-31"},
    ])
    assert res["rows"][0]["value"] == 13


def test_null_hours_never_pass_ordered_filter():
    res = q(metric="count", filters=[{"field": "resolution_hours", "op": "gte", "value": 0}])
    assert res["rows"][0]["value"] == 20  # 24 minus the 4 open cases


def test_malformed_month_is_validation_error_not_silent_zero():
    with pytest.raises(ValidationError, match="YYYY-MM"):
        Query(metric="count", filters=[{"field": "month", "op": "eq", "value": "July 2026"}])


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Query(metric="avg", field="refund_amount")


def test_avg_requires_field():
    with pytest.raises(ValidationError, match="field is required"):
        Query(metric="avg")
