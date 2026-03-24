"""
API contract snapshot tests.

Validates that API response schemas have not changed unexpectedly.
Each test loads a JSON schema snapshot from ./snapshots/ and structurally
validates the live server response — field names, types, and nesting only.
No specific values are asserted.
"""
import json
import os
import pytest
from utils import *

server = ServerPreset.tinyllama2()

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_slots = True
    server.server_metrics = True


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_snapshot(name: str) -> dict:
    path = os.path.join(SNAPSHOT_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


# JSON-Schema-style type names → Python types
_TYPE_MAP = {
    "string":  str,
    "integer": int,
    "number":  (int, float),
    "boolean": bool,
    "array":   list,
    "object":  dict,
}


def _assert_schema(body: dict | list, schema: dict, path: str = "$") -> None:
    """Recursively validate *body* against a simplified JSON-schema *schema*.

    Checks:
      - every key listed in ``required`` exists in *body*
      - every present key whose type is declared in ``properties`` has the
        correct Python type
      - nested ``object`` schemas are validated recursively
      - nested ``array`` schemas validate items against the ``items`` sub-schema
        (first element only — structural check, not exhaustive)
    """
    schema_type = schema.get("type")

    # ── leaf / non-object types ──────────────────────────────────────────
    if schema_type and schema_type != "object":
        expected = _TYPE_MAP.get(schema_type)
        if expected is not None:
            assert isinstance(body, expected), (
                f"{path}: expected type {schema_type}, got {type(body).__name__}"
            )
        return

    # ── object validation ────────────────────────────────────────────────
    assert isinstance(body, dict), (
        f"{path}: expected object (dict), got {type(body).__name__}"
    )

    for key in schema.get("required", []):
        assert key in body, f"{path}: missing required field '{key}'"

    properties = schema.get("properties", {})
    for key, sub_schema in properties.items():
        if key not in body:
            continue  # only validate present keys
        value = body[key]
        child_path = f"{path}.{key}"
        child_type = sub_schema.get("type")

        if child_type == "object":
            _assert_schema(value, sub_schema, child_path)
        elif child_type == "array":
            assert isinstance(value, list), (
                f"{child_path}: expected array, got {type(value).__name__}"
            )
            items_schema = sub_schema.get("items")
            if items_schema and len(value) > 0:
                _assert_schema(value[0], items_schema, f"{child_path}[0]")
        else:
            expected = _TYPE_MAP.get(child_type)
            if expected is not None:
                assert isinstance(value, expected), (
                    f"{child_path}: expected {child_type}, "
                    f"got {type(value).__name__}"
                )


# ── tests ────────────────────────────────────────────────────────────────────

def test_chat_completion_contract():
    """/v1/chat/completions response matches snapshot schema."""
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "Hey"},
        ],
    })
    assert res.status_code == 200
    schema = _load_snapshot("chat_completion")
    _assert_schema(res.body, schema)


def test_completion_contract():
    """/completion response matches snapshot schema."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "I believe the meaning of life is",
    })
    assert res.status_code == 200
    schema = _load_snapshot("completion")
    _assert_schema(res.body, schema)


def test_compact_contract():
    """/compact response matches snapshot schema."""
    global server
    server.start()

    # Fill slot 0 with prompt tokens first
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat named Whiskers who lived in a cozy house",
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200

    # Compact slot 0
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "select",
        "ratio": 2.0,
    })
    assert res.status_code == 200
    schema = _load_snapshot("compact")
    _assert_schema(res.body, schema)


def test_props_contract():
    """/props response matches snapshot schema."""
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    schema = _load_snapshot("props")
    _assert_schema(res.body, schema)


def test_health_contract():
    """/health response matches snapshot schema."""
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200
    schema = _load_snapshot("health")
    _assert_schema(res.body, schema)
