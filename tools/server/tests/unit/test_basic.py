import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_server_start_simple():
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200


def test_server_props():
    global server
    server.offline = False
    server.server_metrics = True
    server.server_slots = True
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    assert res.body["total_slots"] == server.n_slots
    default_val = res.body["default_generation_settings"]
    assert server.n_ctx is not None and server.n_slots is not None
    assert default_val["n_ctx"] == server.n_ctx / server.n_slots
    assert default_val["params"]["seed"] == server.seed

    modelai = res.body["modelai"]
    contract = modelai["contract"]
    caps = modelai["capabilities"]
    runtime = modelai["runtime"]

    assert contract["name"] == "modelai-llama.cpp"
    assert contract["contract_version"] == "0.1.0"
    assert contract["engine_version"]
    assert contract["engine_commit"] != "unknown"
    assert contract["upstream_base_commit"] != "unknown"
    assert contract["server_mode"] == "model"

    assert caps["effective_context_window"] == server.n_ctx / server.n_slots
    assert caps["flash_attention"]["supports_additive_kq_b"] is False
    assert caps["structured_output"]["json_schema"] is True
    assert caps["prompt_cache"]["enabled"] is True
    assert caps["save_restore"]["available"] is True
    assert caps["save_restore"]["validated_for_compacted_path"] is False
    # compaction availability is model-dependent (stories260K is a standard causal
    # model — no SWA, no M-RoPE — so it supports compaction at the runtime level)
    assert isinstance(caps["compacted_prefix"]["available"], bool)
    assert caps["compacted_prefix"]["available"] is True  # stories260K supports it
    assert caps["compacted_prefix"]["enabled"] is False   # no compaction performed yet
    assert isinstance(caps["compacted_prefix"]["flash_attn_overridden"], bool)
    assert isinstance(caps["compacted_prefix"]["last_fallback_reason"], str)
    # tested_envelope (advisory, not enforced — renamed from supported_envelope)
    assert "tested_envelope" in caps["compacted_prefix"]
    envelope = caps["compacted_prefix"]["tested_envelope"]
    assert envelope["min_context"] == 4096
    assert envelope["max_context"] == 16384
    assert envelope["max_ratio_4k"] == 4
    assert "select" in envelope["pipelines"]
    assert caps["features"]["metrics_endpoint"] is True
    assert caps["features"]["slots_endpoint"] is True

    assert runtime["state"] in {"ready", "sleeping"}
    assert runtime["memory"]["allocated_context_bytes"] >= 0
    assert runtime["kv"]["active_n_kv_total"] >= 0
    # runtime compaction section exposes live state, not timing fields
    assert isinstance(runtime["compaction"]["available"], bool)
    assert isinstance(runtime["compaction"]["enabled"], bool)
    assert runtime["compaction"]["available"] is True  # stories260K supports it
    assert runtime["compaction"]["enabled"] is False   # no compaction performed yet


def test_server_models():
    global server
    server.offline = False
    server.server_slots = True
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias

    modelai = res.body["data"][0]["meta"]["modelai"]
    assert modelai["effective_context_window"] == server.n_ctx / server.n_slots
    assert modelai["supports_embeddings"] is False
    assert modelai["supports_json_schema"] is True
    assert modelai["supports_prompt_cache"] is True
    assert modelai["supports_save_restore"] is True
    assert modelai["supports_chat_templates"] is True


def test_server_metrics_contract():
    global server
    server.offline = False
    server.server_metrics = True
    server.server_slots = True
    server.start()
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert "llamacpp:modelai_allocated_model_bytes" in res.body
    assert "llamacpp:modelai_allocated_context_bytes" in res.body
    assert "llamacpp:modelai_active_n_kv_total" in res.body
    assert "llamacpp:modelai_active_n_kv_max" in res.body
    assert "llamacpp:modelai_sequence_state_bytes_total" in res.body
    assert "llamacpp:modelai_prompt_tokens_per_second" in res.body
    assert "llamacpp:modelai_predicted_tokens_per_second" in res.body
    assert "llamacpp:modelai_compacted_prefix_available" in res.body
    assert "llamacpp:modelai_compacted_prefix_enabled" in res.body


def test_server_disabled_capabilities():
    global server
    server.offline = False
    server.server_metrics = False
    server.server_slots = False
    server.start()

    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    caps = props.body["modelai"]["capabilities"]
    assert caps["features"]["metrics_endpoint"] is False
    assert caps["features"]["slots_endpoint"] is False
    assert caps["save_restore"]["available"] is False

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 501
    assert metrics.body["error"]["code"] == 501
    assert metrics.body["error"]["type"] == "not_supported_error"

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 501
    assert slots.body["error"]["code"] == 501
    assert slots.body["error"]["type"] == "not_supported_error"


def test_server_slots():
    global server

    server.offline = False
    server.server_slots = True
    server.n_slots = 2
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert len(res.body) == server.n_slots
    assert server.n_ctx is not None and server.n_slots is not None
    assert res.body[0]["n_ctx"] == server.n_ctx / server.n_slots
    assert "params" not in res.body[0]


def test_load_split_model():
    global server
    server.offline = False
    server.model_hf_repo = "ggml-org/models"
    server.model_hf_file = "tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf"
    server.model_alias = "tinyllama-split"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Hello",
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert match_regex("(little|girl)+", res.body["content"])


def test_no_ui():
    global server
    # default: UI enabled
    server.start()
    url = f"http://{server.server_host}:{server.server_port}"
    res = requests.get(url)
    assert res.status_code == 200
    assert "<!doctype html>" in res.text
    server.stop()

    # with --no-ui, the UI should be disabled
    server.no_ui = True
    server.start()
    res = requests.get(url)
    assert res.status_code == 404


def test_server_model_aliases_and_tags():
    global server
    server.model_alias = "tinyllama-2,fim,code"
    server.model_tags = "chat,fim,small"
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    model = res.body["data"][0]
    # aliases field must contain all aliases
    assert set(model["aliases"]) == {"tinyllama-2", "fim", "code"}
    # tags field must contain all tags
    assert set(model["tags"]) == {"chat", "fim", "small"}
    # id is derived from first alias (alphabetical order from std::set)
    assert model["id"] == "code"
