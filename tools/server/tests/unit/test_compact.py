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
    """POST /compact with solver method works."""
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
    assert res.status_code == 200
    assert res.body["method"] == "solver"
    assert res.body["compacted_tokens"] > 0


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
