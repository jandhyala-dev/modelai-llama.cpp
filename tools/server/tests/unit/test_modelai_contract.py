import re
import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_slots = True
    server.server_metrics = True
    server.temperature = 0.0


# ---------------------------------------------------------------------------
# 1. /props — modelai section structure
# ---------------------------------------------------------------------------

def test_props_modelai_contract_section():
    """/props response contains modelai.contract with name and version."""
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert "modelai" in res.body

    modelai = res.body["modelai"]
    assert "contract" in modelai
    contract = modelai["contract"]
    assert contract["name"] == "modelai-llama.cpp"
    assert isinstance(contract["contract_version"], str)
    assert len(contract["contract_version"]) > 0


def test_props_modelai_capabilities_compacted_prefix():
    """/props response contains modelai.capabilities.compacted_prefix."""
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200

    caps = res.body["modelai"]["capabilities"]
    assert "compacted_prefix" in caps
    cp = caps["compacted_prefix"]
    assert "available" in cp
    assert isinstance(cp["available"], bool)
    assert "enabled" in cp
    assert isinstance(cp["enabled"], bool)


def test_props_modelai_runtime_memory():
    """/props response contains modelai.runtime.memory with allocated_context_bytes."""
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200

    runtime = res.body["modelai"]["runtime"]
    assert "memory" in runtime
    mem = runtime["memory"]
    assert "allocated_context_bytes" in mem
    assert isinstance(mem["allocated_context_bytes"], (int, float))
    assert mem["allocated_context_bytes"] >= 0


def test_props_modelai_runtime_kv():
    """/props response contains modelai.runtime.kv with active_n_kv_total."""
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200

    runtime = res.body["modelai"]["runtime"]
    assert "kv" in runtime
    kv = runtime["kv"]
    assert "active_n_kv_total" in kv
    assert isinstance(kv["active_n_kv_total"], (int, float))
    assert kv["active_n_kv_total"] >= 0


# ---------------------------------------------------------------------------
# 2. /metrics — modelai_* Prometheus metrics
# ---------------------------------------------------------------------------

REQUIRED_MODELAI_METRICS = [
    "llamacpp:modelai_allocated_model_bytes",
    "llamacpp:modelai_allocated_context_bytes",
    "llamacpp:modelai_active_n_kv_total",
    "llamacpp:modelai_active_n_kv_max",
    "llamacpp:modelai_sequence_state_bytes_total",
    "llamacpp:modelai_prompt_tokens_per_second",
    "llamacpp:modelai_predicted_tokens_per_second",
    "llamacpp:modelai_compacted_prefix_available",
    "llamacpp:modelai_compacted_prefix_enabled",
]


def test_metrics_has_all_modelai_metrics():
    """/metrics contains every required modelai_* Prometheus metric."""
    global server
    server.start()
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    metrics_text = res.body if isinstance(res.body, str) else str(res.body)
    for metric_name in REQUIRED_MODELAI_METRICS:
        assert metric_name in metrics_text, f"missing metric: {metric_name}"


def test_metrics_modelai_values_are_numeric():
    """/metrics modelai_* metric values are valid numbers (not NaN, not empty)."""
    global server
    server.start()
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    metrics_text = res.body if isinstance(res.body, str) else str(res.body)

    for metric_name in REQUIRED_MODELAI_METRICS:
        # Match lines like: llamacpp:modelai_foo 123.45
        # Skip comment/type lines starting with #
        pattern = rf'^{re.escape(metric_name)}\s+([\d.eE+\-]+)'
        match = re.search(pattern, metrics_text, re.MULTILINE)
        assert match is not None, f"no numeric value found for {metric_name}"
        value_str = match.group(1)
        value = float(value_str)
        assert not (value != value), f"{metric_name} has NaN value"  # NaN check


# ---------------------------------------------------------------------------
# 3. /compact — success path (needs prompt loaded in slot)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_compact_returns_valid_response():
    """POST /compact with valid params returns 200 with compaction_time_ms > 0."""
    global server
    server.start()

    # Fill slot 0 with prompt tokens
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
    body = res.body
    assert body["compaction_time_ms"] > 0
    assert body["original_tokens"] > body["compacted_tokens"]


def test_compact_empty_slot_returns_4xx():
    """POST /compact on an empty slot returns 4xx, not 500."""
    global server
    server.start()

    # No prompt loaded — compact on empty slot
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "select",
        "ratio": 2.0,
    })
    # Must be a client error (4xx), not a server error (5xx)
    assert 400 <= res.status_code < 500, \
        f"expected 4xx for empty slot, got {res.status_code}"
    assert "error" in res.body


# ---------------------------------------------------------------------------
# 4. /compact — invalid slot returns 400, not 500
# ---------------------------------------------------------------------------

def test_compact_invalid_slot_returns_400():
    """POST /compact with non-existent slot ID returns 400, not 500."""
    global server
    server.start()

    res = server.make_request("POST", "/compact", data={
        "id_slot": 999,
        "method": "select",
        "ratio": 2.0,
    })
    assert 400 <= res.status_code < 500, \
        f"expected 4xx for invalid slot, got {res.status_code}"
    assert "error" in res.body


def test_compact_negative_slot_returns_400():
    """POST /compact with negative slot ID returns 400, not 500."""
    global server
    server.start()

    res = server.make_request("POST", "/compact", data={
        "id_slot": -1,
        "method": "select",
        "ratio": 2.0,
    })
    assert 400 <= res.status_code < 500, \
        f"expected 4xx for negative slot, got {res.status_code}"
    assert "error" in res.body
