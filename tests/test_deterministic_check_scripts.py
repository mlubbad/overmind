"""Regression tests for built-in deterministic span check scripts."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.covers("metrics-math")

SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent / "overbae" / "innermind" / "checks" / "scripts"
)

ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "AttributeError": AttributeError,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "IndexError": IndexError,
    "int": int,
    "isinstance": isinstance,
    "KeyError": KeyError,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "RuntimeError": RuntimeError,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}


def _load_script(name: str, **overrides):
    source = (SCRIPTS_DIR / f"{name}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.Import, ast.ImportFrom)), f"{name} must stay import-free"
    namespace = {"__builtins__": ALLOWED_BUILTINS, "json": json, "re": re}
    exec(compile(tree, f"{name}.py", "exec"), namespace)
    namespace.update(overrides)
    return namespace["evaluate"]


def test_schema_field_conformance_enforces_non_string_enums_without_bool_int_confusion():
    evaluate = _load_script(
        "schema_field_conformance",
        FIELDS={
            "priority": {"type": "integer", "enum": [1, 2, 3]},
            "approved": {"type": "boolean", "enum": [True]},
            "status": {"type": "string", "enum": ["open", "closed"]},
        },
    )

    score, reason = evaluate(
        SimpleNamespace(output='{"priority": 999, "approved": true, "status": "open"}')
    )
    assert score == 2 / 3
    assert "priority" in reason

    score_ok, _ = evaluate(
        SimpleNamespace(output='{"priority": 2, "approved": true, "status": "CLOSED"}')
    )
    assert score_ok == 1.0

    # bool True must not satisfy integer enum [1] and int 1 must not satisfy boolean enum [True]
    eval_strict = _load_script(
        "schema_field_conformance",
        FIELDS={"priority": {"enum": [1]}, "flag": {"enum": [True]}},
    )
    score_mismatch, _ = eval_strict(SimpleNamespace(output='{"priority": true, "flag": 1}'))
    assert score_mismatch == 0.0


def test_tool_args_match_requires_key_presence_for_none_values():
    evaluate_subset = _load_script(
        "tool_args_match",
        EXPECTED_CALLS=({"name": "search", "arguments": {"filter": None}},),
        MODE="subset",
    )
    missing_key_span = SimpleNamespace(
        children=[
            SimpleNamespace(
                span_type="tool_call",
                tool="search",
                name="search",
                attributes={},
                input="{}",
                children=[],
            )
        ]
    )
    present_none_span = SimpleNamespace(
        children=[
            SimpleNamespace(
                span_type="tool_call",
                tool="search",
                name="search",
                attributes={},
                input='{"filter": null}',
                children=[],
            )
        ]
    )

    score_missing, _ = evaluate_subset(missing_key_span)
    assert score_missing == 0.0

    score_present, _ = evaluate_subset(present_none_span)
    assert score_present == 1.0

    evaluate_superset = _load_script(
        "tool_args_match",
        EXPECTED_CALLS=({"name": "search", "arguments": {}},),
        MODE="superset",
    )
    score_extra_none, _ = evaluate_superset(present_none_span)
    assert score_extra_none == 0.0


def test_exact_match_strips_both_output_and_reference():
    evaluate = _load_script("exact_match", REFERENCE="  approved  ")
    passed, _ = evaluate(SimpleNamespace(output="approved\n"))
    assert passed is True


def test_latency_budget_passes_zero_duration_instead_of_abstaining():
    evaluate = _load_script("latency_budget", MAX_SECONDS=1.0)
    passed, reason = evaluate(SimpleNamespace(duration_s=0.0))
    assert passed is True
    assert "0.0s" in reason
    assert evaluate(SimpleNamespace(duration_s=None)) is None
    assert evaluate(SimpleNamespace(duration_s=True)) is None


def test_cost_budget_rejects_boolean_values():
    evaluate = _load_script("cost_budget", MAX_COST=2.0)
    assert evaluate(SimpleNamespace(usage={"genai.cost": True})) is None
    passed, _ = evaluate(SimpleNamespace(usage={"genai.cost": 0.0}))
    assert passed is True
