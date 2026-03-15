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


def test_compact_after_chat_completions():
    """5.2.1: Compact after /v1/chat/completions."""
    global server
    server.start()
    # Multi-turn chat
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [
            {"role": "user", "content": "Tell me a story about a cat."},
        ],
        "max_tokens": 16,
    })
    assert res.status_code == 200
    # Compact
    res = server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "select", "ratio": 2.0,
    })
    assert res.status_code == 200
    assert res.body["compacted_tokens"] > 0


def test_compact_continue_after_chat():
    """5.2.2: Continue chat after compaction."""
    global server
    server.start()
    # Initial chat
    server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "Tell me a story about a cat."}],
        "max_tokens": 16,
    })
    # Compact
    server.make_request("POST", "/compact", data={
        "id_slot": 0, "method": "select", "ratio": 2.0,
    })
    # Continue chat
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [
            {"role": "user", "content": "Tell me a story about a cat."},
            {"role": "assistant", "content": "Once upon a time"},
            {"role": "user", "content": "What happened next?"},
        ],
        "max_tokens": 16,
    })
    assert res.status_code == 200
    assert len(res.body["choices"]) > 0
