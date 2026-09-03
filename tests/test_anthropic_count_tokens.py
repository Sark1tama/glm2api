import io
import json
from types import SimpleNamespace

import pytest

from glm2api.api import server as server_module
from glm2api.api.adapters.anthropic.messages import (
    count_anthropic_cache_control_markers,
    estimate_anthropic_input_tokens,
)


def test_anthropic_input_estimate_reuses_shared_rule_and_ignores_controls():
    base = {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": "检查仓库"}],
    }
    with_controls = {
        **base,
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0,
        "top_p": 0.9,
        "stop_sequences": ["DONE"],
        "metadata": {"user_id": "not-prompt"},
    }
    with_context = {
        **base,
        "system": [{"type": "text", "text": "遵守安全规范"}],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "检查仓库"},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "cGRm",
                        },
                    },
                ],
            }
        ],
    }

    assert estimate_anthropic_input_tokens(base) == estimate_anthropic_input_tokens(with_controls)
    assert estimate_anthropic_input_tokens(with_context) > estimate_anthropic_input_tokens(base)


def test_anthropic_cache_control_markers_are_counted_without_changing_payload():
    payload = {
        "system": [{"type": "text", "text": "规则", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请求", "cache_control": {"type": "ephemeral"}},
                    {"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}},
                ],
            }
        ],
        "tools": [{"name": "Bash", "cache_control": {"type": "ephemeral"}}],
    }

    assert count_anthropic_cache_control_markers(payload) == 3


def test_anthropic_count_tokens_route_estimates_without_calling_glm():
    class NoUpstreamCall:
        def __getattr__(self, name):
            raise AssertionError(f"count_tokens must not call GLM: {name}")

    config = SimpleNamespace(
        api_prefix="/v1",
        cors_allow_origin="*",
        server_api_keys=[],
        debug_dump_all=False,
    )
    payload = {
        "model": "glm-5.3",
        "system": [{"type": "text", "text": "你是代码助手"}],
        "messages": [{"role": "user", "content": "检查仓库"}],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    }

    handler = _build_post_handler(
        path="/v1/messages/count_tokens?beta=true",
        payload=payload,
        glm_client=NoUpstreamCall(),
        config=config,
    )
    handler.do_POST()
    result = json.loads(handler.wfile.getvalue().decode("utf-8"))

    assert result == {"input_tokens": estimate_anthropic_input_tokens(payload)}


def test_anthropic_count_tokens_invalid_payload_uses_anthropic_error_envelope():
    config = SimpleNamespace(
        api_prefix="/v1",
        cors_allow_origin="*",
        server_api_keys=[],
        debug_dump_all=False,
    )
    handler = _build_post_handler(
        path="/v1/messages/count_tokens",
        payload={"model": "glm-5.3"},
        glm_client=object(),
        config=config,
    )
    handler.do_POST()

    result = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.sent_statuses == [400]
    assert result["type"] == "error"
    assert result["error"]["type"] == "invalid_request_error"


def test_anthropic_count_tokens_requires_anthropic_auth_when_configured():
    config = SimpleNamespace(
        api_prefix="/v1",
        cors_allow_origin="*",
        server_api_keys=["secret"],
        debug_dump_all=False,
    )
    handler = _build_post_handler(
        path="/v1/messages/count_tokens",
        payload={
            "model": "glm-5.3",
            "messages": [{"role": "user", "content": "hi"}],
        },
        glm_client=object(),
        config=config,
    )
    handler.do_POST()

    result = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.sent_statuses == [401]
    assert result["type"] == "error"
    assert result["error"]["type"] == "authentication_error"


def _build_post_handler(path, payload, glm_client, config):
    class FakeLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    server = server_module.GLM2APIServer.__new__(server_module.GLM2APIServer)
    server.config = config
    server.glm_client = glm_client
    server.logger = FakeLogger()
    handler_cls = server._build_handler()
    handler = object.__new__(handler_cls)
    body = json.dumps(payload).encode("utf-8")
    handler.command = "POST"
    handler.path = path
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.sent_statuses = []
    handler.sent_headers = {}
    handler.send_response = handler.sent_statuses.append
    handler.send_header = handler.sent_headers.__setitem__
    handler.end_headers = lambda *args, **kwargs: None
    return handler
