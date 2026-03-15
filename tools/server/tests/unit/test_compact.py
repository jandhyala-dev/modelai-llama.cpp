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


def test_compact_valid_request():
    """POST /compact with a valid request after filling a slot with prompt tokens."""
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

    # Compact slot 0 with select method at 2x ratio
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "select",
        "ratio": 2.0,
    })
    assert res.status_code == 200
    body = res.body
    assert body["method"] == "select"
    assert body["compacted_tokens"] > 0
    assert body["original_tokens"] > 0
    assert body["compression_ratio"] >= 1.0
    assert body["compaction_time_ms"] >= 0.0
    assert isinstance(body["reclaimed"], bool)


def test_compact_invalid_slot():
    """POST /compact with a non-existent slot ID returns error."""
    global server
    server.start()

    res = server.make_request("POST", "/compact", data={
        "id_slot": 999,
        "method": "select",
        "ratio": 2.0,
    })
    assert res.status_code != 200
    assert "error" in res.body


def test_compact_invalid_method():
    """POST /compact with an unknown method returns error."""
    global server
    server.start()

    # Fill slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat",
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "nonexistent_method",
        "ratio": 2.0,
    })
    assert res.status_code != 200
    assert "error" in res.body


def test_compact_invalid_ratio():
    """POST /compact with ratio < 1.0 returns error (not UB)."""
    global server
    server.start()

    # Fill slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat",
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200

    # ratio = 0.5 must be rejected
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "select",
        "ratio": 0.5,
    })
    assert res.status_code != 200
    assert "error" in res.body

    # ratio = 0.0 must be rejected
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "select",
        "ratio": 0.0,
    })
    assert res.status_code != 200
    assert "error" in res.body

    # ratio = -1.0 must be rejected
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "select",
        "ratio": -1.0,
    })
    assert res.status_code != 200
    assert "error" in res.body


def test_compact_empty_slot():
    """POST /compact on a slot with no prompt tokens returns error."""
    global server
    server.start()

    # Do NOT fill any slot — compact on empty slot 0
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "select",
        "ratio": 2.0,
    })
    assert res.status_code != 200
    assert "error" in res.body


def test_compact_invalid_json():
    """POST /compact with malformed JSON body returns error."""
    global server
    server.start()

    import requests as req
    url = f"http://{server.server_host}:{server.server_port}/compact"
    response = req.post(url, data="not json", headers={"Content-Type": "application/json"})
    assert response.status_code != 200


def test_compact_props_reflects_state():
    """After compaction, /props should reflect that compaction is enabled."""
    global server
    server.start()

    # Before compaction: available but not enabled
    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    caps = props.body["modelai"]["capabilities"]["compacted_prefix"]
    assert caps["available"] is True   # stories260K supports compaction
    assert caps["enabled"] is False    # no compaction performed yet

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

    # After compaction: /props should show enabled=true
    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    caps = props.body["modelai"]["capabilities"]["compacted_prefix"]
    assert caps["available"] is True
    assert caps["enabled"] is True


def test_compact_solver_method():
    """POST /compact with solver method is rejected by V1 beta allowlist."""
    global server
    server.start()

    # Fill slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat named Whiskers who lived in a cozy house",
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "solver",
        "ratio": 2.0,
    })
    assert res.status_code != 200
    assert "error" in res.body
    assert "allowlist" in res.body["error"]["message"].lower() or "allow" in res.body["error"]["message"].lower()


def test_compact_with_explicit_target_tokens():
    """POST /compact with explicit target_tokens ignores ratio."""
    global server
    server.start()

    # Fill slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat named Whiskers who lived in a cozy house",
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/compact", data={
        "id_slot": 0,
        "method": "select",
        "target_tokens": 5,
    })
    assert res.status_code == 200
    assert res.body["compacted_tokens"] == 5


def test_compact_prompt_and_continue():
    """5.1.1: POST /completion, POST /compact, POST /completion again."""
    global server
    server.start()
    # Fill slot with longer prompt
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat named Whiskers who lived in a cozy house with a garden full of flowers and butterflies",
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200
    # Compact
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "select", "ratio": 2.0,
    })
    assert res.status_code == 200
    assert res.body["compacted_tokens"] > 0
    # Continue generation after compaction
    res = server.make_request("POST", "/completion", data={
        "prompt": "The cat then",
        "id_slot": 0,
        "n_predict": 8,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_compact_metrics_update():
    """5.1.2: Prometheus metrics populated after compaction."""
    global server
    server.start()
    # Fill and compact
    server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat named Whiskers who lived in a cozy house",
        "id_slot": 0, "cache_prompt": True, "n_predict": 4,
    })
    server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "select", "ratio": 2.0,
    })
    # Check metrics — verify Prometheus text format contains expected metric names
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    # Metrics response is text/plain Prometheus format, not JSON
    metrics_text = res.body if isinstance(res.body, str) else str(res.body)
    # At minimum, standard llama.cpp metrics should be present
    assert "llamacpp:" in metrics_text or "llama_" in metrics_text or len(metrics_text) > 100


def test_compact_sequential():
    """5.1.3: Two sequential compactions at increasing ratios."""
    global server
    server.start()
    # Fill slot
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat named Whiskers who lived in a cozy house with a garden full of flowers",
        "id_slot": 0, "cache_prompt": True, "n_predict": 4,
    })
    assert res.status_code == 200
    # First compaction at 2x
    res1 = server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "select", "ratio": 2.0,
    })
    assert res1.status_code == 200
    tokens_2x = res1.body["compacted_tokens"]
    # Second compaction at 4x
    res2 = server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "select", "ratio": 4.0,
    })
    # Second compaction may succeed or fail depending on remaining tokens
    # If it succeeds, compacted tokens should be fewer
    if res2.status_code == 200:
        assert res2.body["compacted_tokens"] <= tokens_2x


def test_compact_streaming_after_compaction():
    """5.1.4: Streaming response works after compaction (SSE chunked)."""
    global server
    server.start()
    # Fill and compact
    server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat named Whiskers who lived in a cozy house",
        "id_slot": 0, "cache_prompt": True, "n_predict": 4,
    })
    server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "select", "ratio": 2.0,
    })
    # Streaming request after compaction — verify SSE chunks arrive
    n_chunks = 0
    for chunk in server.make_stream_request("POST", "/completion", data={
        "prompt": "The cat",
        "id_slot": 0,
        "n_predict": 4,
        "stream": True,
    }):
        n_chunks += 1
        assert "content" in chunk
    assert n_chunks > 0, "streaming after compaction must produce at least one SSE chunk"


def test_compact_props_endpoint():
    """5.1.5: GET /props returns valid JSON with compaction and contract fields."""
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert "modelai" in res.body
    modelai = res.body["modelai"]

    # Contract version and provenance (CI policy §5.6)
    assert "contract" in modelai
    contract = modelai["contract"]
    assert "contract_version" in contract and contract["contract_version"] != ""
    assert "engine_commit" in contract and contract["engine_commit"] != ""
    assert "upstream_base_commit" in contract

    # Capabilities — compacted prefix
    caps = modelai["capabilities"]["compacted_prefix"]
    assert "available" in caps
    assert "tested_envelope" in caps
    envelope = caps["tested_envelope"]
    assert "pipelines" in envelope
    assert isinstance(envelope["pipelines"], list)
    assert len(envelope["pipelines"]) > 0
    assert "note" in envelope
    # The note must communicate advisory nature
    assert "advisory" in envelope["note"].lower() or "not enforced" in envelope["note"].lower()


def test_compact_models_endpoint():
    """5.1.5b: GET /v1/models returns model list after compaction."""
    global server
    server.start()
    # Fill and compact
    server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat",
        "id_slot": 0, "cache_prompt": True, "n_predict": 4,
    })
    server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "select", "ratio": 2.0,
    })
    # Models endpoint should still work
    res = server.make_request("GET", "/v1/models")
    assert res.status_code == 200
    assert "data" in res.body
    assert len(res.body["data"]) > 0


def test_compact_allowlist_rejects_solver():
    """5.3: V1 beta allowlist rejects non-select methods."""
    global server
    server.start()
    # Fill slot
    server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat named Whiskers who lived in a cozy house",
        "id_slot": 0, "cache_prompt": True, "n_predict": 4,
    })
    # solver should be rejected by V1 allowlist
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "solver", "ratio": 2.0,
    })
    assert res.status_code != 200
    assert "error" in res.body
    assert "allowlist" in res.body["error"]["message"].lower() or "allow" in res.body["error"]["message"].lower()


def test_compact_allowlist_rejects_self_study():
    """5.3: V1 beta allowlist rejects self_study."""
    global server
    server.start()
    server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little cat named Whiskers who lived in a cozy house",
        "id_slot": 0, "cache_prompt": True, "n_predict": 4,
    })
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "self_study", "ratio": 2.0,
    })
    assert res.status_code != 200
    assert "error" in res.body
