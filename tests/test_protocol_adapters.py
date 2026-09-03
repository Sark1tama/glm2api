import io
import json
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from glm2api.api import server as server_module
from glm2api.api.adapters.anthropic.messages import (
    AnthropicMessagesStreamAccumulator,
    anthropic_messages_to_internal,
    internal_to_anthropic_messages_response,
)
from glm2api.glm.auth import GLMAccessTokenManager
from glm2api.glm.client import ConcurrentRequestQueue, GLMWebClient, QueueLease, UpstreamAPIError
from glm2api.glm.errors import QueueTimeoutError
from glm2api.core.models import (
    ContentBlock,
    Message,
    TextGenerationRequest,
    TextGenerationResponse,
    TextStreamEvent,
    ToolCallDelta,
    ToolChoice,
    ToolDefinition,
)
from glm2api.core.usage import TokenUsage

from glm2api.api.adapters.openai.responses import (
    OpenAIResponsesStreamAccumulator,
    internal_to_openai_responses_response,
    openai_responses_to_internal,
)


class _DummyConfig:
    glm_user_agent = "Mozilla/5.0"


def test_request_queue_skips_timed_out_ticket():
    queue = ConcurrentRequestQueue(
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        wait_timeout=0.01,
        max_concurrency=1,
    )
    first = queue.acquire("first")
    with pytest.raises(QueueTimeoutError):
        queue.acquire("timed-out")

    first.release()
    next_lease = queue.acquire("next")
    next_lease.release()


def test_get_browser_headers_includes_random_x_forwarded_for():
    manager = GLMAccessTokenManager.__new__(GLMAccessTokenManager)
    manager.config = _DummyConfig()

    headers = manager.get_browser_headers()
    xff = headers["X-Forwarded-For"]
    octets = xff.split(".")

    assert len(octets) == 4
    assert all(part.isdigit() for part in octets)
    assert 1 <= int(octets[0]) <= 223
    assert int(octets[0]) not in {10, 127, 169, 172, 192}


def test_openai_responses_to_internal_preserves_tool_choice():
    payload = {
        "model": "glm-4",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "查询天气",
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }

    converted = openai_responses_to_internal(payload)

    assert converted.tool_choice is not None
    assert converted.tool_choice.mode == "function"
    assert converted.tool_choice.name == "get_weather"


def test_glm_client_resolves_tool_choice_before_prompt_translation():
    client = GLMWebClient.__new__(GLMWebClient)
    client.config = SimpleNamespace(blocked_tool_names=[])
    client.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    tools = (
        ToolDefinition(name="get_weather"),
        ToolDefinition(name="calculate"),
    )

    none_tools, none_names = client._resolve_tools(
        TextGenerationRequest(model="glm-4", messages=(), tools=tools, tool_choice=ToolChoice(mode="none"))
    )
    assert none_tools is None
    assert none_names == set()

    selected_tools, selected_names = client._resolve_tools(
        TextGenerationRequest(
            model="glm-4",
            messages=(),
            tools=tools,
            tool_choice=ToolChoice(mode="function", name="get_weather"),
        )
    )
    assert selected_tools == list(tools)
    assert selected_names == {"get_weather", "calculate"}

    with pytest.raises(ValueError, match="不可用"):
        client._resolve_tools(
            TextGenerationRequest(
                model="glm-4",
                messages=(),
                tools=tools,
                tool_choice=ToolChoice(mode="function", name="missing"),
            )
        )

    with pytest.raises(ValueError, match="没有可用工具"):
        client._resolve_tools(
            TextGenerationRequest(
                model="glm-4",
                messages=(),
                tool_choice=ToolChoice(mode="required"),
            )
        )


def test_glm_client_rejects_generation_parameters_without_web_equivalent():
    client = GLMWebClient.__new__(GLMWebClient)
    client.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

    for field, value in (
        ("temperature", 0.2),
        ("top_p", 0.9),
        ("stop", ("END",)),
        ("response_format", {"type": "json_object"}),
    ):
        request = TextGenerationRequest(model="glm-4", messages=())
        setattr(request, field, value)
        with pytest.raises(ValueError, match="不支持生成参数"):
            client._validate_generation_parameters(request)


def test_glm_client_warns_when_max_tokens_cannot_reach_web_protocol():
    warnings = []
    client = GLMWebClient.__new__(GLMWebClient)
    client.logger = SimpleNamespace(warning=lambda message, *args, **kwargs: warnings.append(message))

    client._validate_generation_parameters(
        TextGenerationRequest(model="glm-4", messages=(), max_tokens=128)
    )

    assert warnings
    assert "max_tokens" in warnings[0]


def test_glm_client_applies_model_specific_multimodal_capability():
    client = GLMWebClient.__new__(GLMWebClient)
    image_request = TextGenerationRequest(
        model="glm-5.3",
        messages=(
            Message(
                role="user",
                content=(
                    ContentBlock(kind="text", text="看图"),
                    ContentBlock(kind="image", url="data:image/png;base64,abc"),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"模型 glm-5\.3 仅支持文本输入.*content\[1\]"):
        client._validate_model_content(image_request, image_request.model)

    file_request = TextGenerationRequest(
        model="glm-5.3",
        messages=(
            Message(
                role="user",
                content=(ContentBlock(kind="file", url="data:application/pdf;base64,AQI="),),
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"模型 glm-5\.3 仅支持文本输入.*content\[0\]"):
        client._validate_model_content(file_request, file_request.model)
    client._validate_model_content(file_request, "glm-5.3-flash")

    with pytest.raises(ValueError, match=r"模型 text-alias 仅支持文本输入"):
        client._validate_model_content(image_request, "text-alias", "glm-5.3")

    flash_request = TextGenerationRequest(
        model="glm-5.3-flash",
        messages=image_request.messages,
    )
    client._validate_model_content(flash_request, flash_request.model)

    thinking_request = TextGenerationRequest(
        model="glm-5.3",
        messages=(
            Message(
                role="assistant",
                content=(
                    ContentBlock(kind="thinking", text="摘要", metadata={"signature": "sig_1"}),
                    ContentBlock(kind="redacted_thinking", metadata={"data": "opaque_1"}),
                ),
            ),
        ),
    )
    client._validate_model_content(thinking_request, thinking_request.model)


def test_glm_client_request_body_maps_supported_generation_switches(monkeypatch):
    captured = {}

    class FakeLogger:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass

    class FakeResponse:
        pass

    client = GLMWebClient.__new__(GLMWebClient)
    client.config = SimpleNamespace(
        blocked_tool_names=[],
        debug_dump_all=False,
        model_aliases={},
        glm_assistant_id="assistant_1",
        chat_stream_url="https://example.test/stream",
        request_timeout=1,
        glm_busy_max_retries=0,
        glm_busy_retry_interval=0,
    )
    client.logger = FakeLogger()
    client.auth = SimpleNamespace(get_browser_headers=lambda: {})
    client.files = SimpleNamespace(upload_referenced_files=lambda messages: [])
    client._prepare_chat_response = lambda response: response
    client.call_with_account_failover = (
        lambda request_name, operation, preferred_account_index=None: operation(0, "token")
    )

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("glm2api.glm.client.build_sign", lambda: ("ts", "nonce", "sign"))
    monkeypatch.setattr("glm2api.glm.client.urllib.request.urlopen", fake_urlopen)

    client._open_chat_stream(
        TextGenerationRequest(
            model="glm-5.3",
            messages=(Message(role="user", content="联网查一下"),),
            reasoning_effort="high",
            web_search=True,
        )
    )

    metadata = captured["body"]["meta_data"]
    assert metadata["chat_mode"] == "deep_thinking"
    assert metadata["is_networking"] is True
    assert "temperature" not in metadata
    assert "top_p" not in metadata
    assert "max_tokens" not in metadata


def test_glm_signed_request_composes_shared_headers(monkeypatch):
    client = GLMWebClient.__new__(GLMWebClient)
    client.auth = SimpleNamespace(
        get_browser_headers=lambda app_fr="browser_extension": {"X-App-Fr": app_fr}
    )

    monkeypatch.setattr("glm2api.glm.client.build_sign", lambda: ("ts", "nonce", "sign"))

    request = client.build_signed_request(
        "https://example.test/upload",
        method="POST",
        access_token="access",
        app_fr="default",
        referer="https://chatglm.cn/video?lang=zh",
        extra_headers={"Content-Type": "multipart/form-data"},
    )
    headers = {key.lower(): value for key, value in request.header_items()}

    assert headers["x-app-fr"] == "default"
    assert headers["authorization"] == "Bearer access"
    assert headers["referer"] == "https://chatglm.cn/video?lang=zh"
    assert headers["content-type"] == "multipart/form-data"
    assert headers["x-nonce"] == "nonce"
    assert headers["x-sign"] == "sign"


def test_glm_upstream_request_normalizes_http_error(monkeypatch):
    client = GLMWebClient.__new__(GLMWebClient)
    client.config = SimpleNamespace(request_timeout=1)
    error = urllib.error.HTTPError(
        "https://example.test/stream",
        429,
        "busy",
        {"Content-Type": "application/json"},
        io.BytesIO(b'{"status":10061,"message":"busy"}'),
    )

    def fail_urlopen(request, timeout):
        raise error

    monkeypatch.setattr("glm2api.glm.client.urllib.request.urlopen", fail_urlopen)

    with pytest.raises(UpstreamAPIError, match="HTTP 429") as raised:
        client.open_upstream_request(urllib.request.Request("https://example.test/stream"))

    assert raised.value.status_code == 429
    assert raised.value.payload == {"status": 10061, "message": "busy"}


def test_protocol_adapters_return_internal_text_requests():
    anthropic_request = anthropic_messages_to_internal(
        {
            "model": "glm-4",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
    )
    responses_request = openai_responses_to_internal(
        {"model": "glm-4", "input": [{"role": "user", "content": "hi"}]}
    )

    assert isinstance(anthropic_request, TextGenerationRequest)
    assert isinstance(anthropic_request.messages[0].content, str)
    assert isinstance(responses_request, TextGenerationRequest)
    assert responses_request.messages[0].role == "user"


def test_anthropic_messages_to_internal_preserves_mixed_tool_result_order():
    converted = anthropic_messages_to_internal(
        {
            "model": "glm-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "工具返回如下："},
                        {"type": "tool_result", "tool_use_id": "call_1", "content": "晴"},
                        {"type": "text", "text": "请继续"},
                    ],
                },
            ],
        }
    )

    assert [message.role for message in converted.messages] == ["assistant", "user", "tool", "user"]
    assert converted.messages[0].tool_calls[0].id == "call_1"
    assert converted.messages[1].content == "工具返回如下："
    assert converted.messages[2].tool_call_id == "call_1"
    assert converted.messages[2].content == "晴"
    assert converted.messages[3].content == "请继续"


def test_anthropic_messages_to_internal_preserves_assistant_text_tool_text_order():
    converted = anthropic_messages_to_internal(
        {
            "model": "glm-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "先说明"},
                        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}},
                        {"type": "text", "text": "后续"},
                    ],
                },
            ],
        }
    )

    assert [message.role for message in converted.messages] == ["assistant", "assistant", "assistant"]
    assert converted.messages[0].content == "先说明"
    assert converted.messages[1].tool_calls[0].id == "call_1"
    assert converted.messages[2].content == "后续"


def test_anthropic_messages_to_internal_accepts_claude_code_style_history():
    converted = anthropic_messages_to_internal(
        {
            "model": "glm-4",
            "system": [
                {
                    "type": "text",
                    "text": "Follow the user's instructions.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "input_schema": {"type": "object"},
                }
            ],
            "messages": [
                {"role": "user", "content": "检查仓库"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "先查看状态", "signature": "sig"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Bash",
                            "input": {"command": "git status --short"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "干净"}],
                        }
                    ],
                },
            ],
        }
    )

    assert converted.messages[0] == Message(role="system", content="Follow the user's instructions.")
    assert converted.messages[3].tool_calls[0].name == "Bash"
    assert converted.messages[3].tool_calls[0].arguments == '{"command":"git status --short"}'
    assert converted.messages[4] == Message(role="tool", content="干净", tool_call_id="toolu_1")


def test_anthropic_messages_to_internal_preserves_image_tool_result_content():
    converted = anthropic_messages_to_internal(
        {
            "model": "glm-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "screenshot", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "截图"},
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    result_content = converted.messages[1].content
    assert isinstance(result_content, tuple)
    assert result_content[0].text == "截图"
    assert result_content[1].url == "data:image/png;base64,abc"


@pytest.mark.parametrize("block_type", ["server_tool_use", "web_search_tool_result"])
def test_anthropic_messages_to_internal_rejects_unsupported_content_blocks(block_type):
    with pytest.raises(ValueError, match="暂不支持"):
        anthropic_messages_to_internal(
            {
                "model": "glm-4",
                "messages": [
                    {"role": "user", "content": [{"type": block_type}]},
                ],
            }
        )


def test_anthropic_adapter_rejects_malformed_messages_instead_of_dropping_them():
    with pytest.raises(ValueError, match="messages 必须是数组"):
        anthropic_messages_to_internal({"model": "glm-4", "messages": "invalid"})
    with pytest.raises(ValueError, match="name 不能为空"):
        anthropic_messages_to_internal(
            {
                "model": "glm-4",
                "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "input": {}}]}],
            }
        )


def test_anthropic_adapter_accepts_max_output_tokens_alias():
    converted = anthropic_messages_to_internal(
        {
            "model": "glm-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_output_tokens": 64,
        }
    )
    assert converted.max_tokens == 64


def test_anthropic_document_sources_become_uploadable_file_blocks():
    converted = anthropic_messages_to_internal(
        {
            "model": "glm-5.3-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "AQI=",
                            },
                            "title": "报告",
                            "context": "只关注风险章节",
                        },
                        {
                            "type": "document",
                            "source": {"type": "url", "url": "https://example.test/report.pdf"},
                        },
                    ],
                }
            ],
        }
    )

    content = converted.messages[0].content
    assert isinstance(content, tuple)
    assert content[0] == ContentBlock(kind="file", url="data:application/pdf;base64,AQI=")
    assert content[1] == ContentBlock(kind="text", text="Document title: 报告\n只关注风险章节")
    assert content[2] == ContentBlock(kind="file", url="https://example.test/report.pdf")


def test_anthropic_document_text_and_content_sources_are_preserved():
    converted = anthropic_messages_to_internal(
        {
            "model": "glm-5.3-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "document", "source": {"type": "text", "data": "纯文本文档"}},
                        {
                            "type": "document",
                            "source": {
                                "type": "content",
                                "content": [{"type": "text", "text": "第二段"}],
                            },
                        },
                    ],
                }
            ],
        }
    )

    content = converted.messages[0].content
    assert isinstance(content, tuple)
    assert content == (
        ContentBlock(kind="text", text="纯文本文档"),
        ContentBlock(kind="text", text="第二段"),
    )


def test_anthropic_document_file_id_requires_a_local_file_reference():
    with pytest.raises(ValueError, match=r"file_id.*base64、url、text 或 content"):
        anthropic_messages_to_internal(
            {
                "model": "glm-5.3-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "document", "source": {"type": "file", "file_id": "file_123"}},
                        ],
                    }
                ],
            }
        )


def test_anthropic_messages_to_internal_preserves_thinking_and_redacted_blocks():
    converted = anthropic_messages_to_internal(
        {
            "model": "glm-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "先分析", "signature": "sig_1"},
                        {"type": "redacted_thinking", "data": "opaque_1"},
                        {"type": "text", "text": "继续"},
                    ],
                }
            ],
        }
    )

    message = converted.messages[0]
    assert isinstance(message.content, tuple)
    assert message.content[0] == ContentBlock(
        kind="thinking",
        text="先分析",
        metadata={"signature": "sig_1"},
    )
    assert message.content[1] == ContentBlock(
        kind="redacted_thinking",
        metadata={"data": "opaque_1"},
    )


def test_anthropic_messages_to_internal_rejects_empty_redacted_thinking_data():
    with pytest.raises(ValueError, match=r"content\[0\]\.data 不能为空"):
        anthropic_messages_to_internal(
            {
                "model": "glm-4",
                "messages": [
                    {"role": "assistant", "content": [{"type": "redacted_thinking", "data": ""}]},
                ],
            }
        )


def test_internal_to_anthropic_omits_unsigned_reasoning_and_keeps_signed_blocks():
    response = internal_to_anthropic_messages_response(
        TextGenerationResponse(
            response_id="chat_1",
            model="glm-4",
            created=1,
            message=Message(
                role="assistant",
                content=(
                    ContentBlock(kind="thinking", text="signed", metadata={"signature": "sig_1"}),
                    ContentBlock(kind="redacted_thinking", metadata={"data": "opaque_1"}),
                    ContentBlock(kind="text", text="完成"),
                ),
                reasoning_content="GLM internal reasoning",
            ),
            finish_reason="stop",
            usage=TokenUsage.estimated(input_tokens=2, output_tokens=3),
        ),
        model="glm-4",
    )

    assert response["content"] == [
        {"signature": "sig_1", "type": "thinking", "thinking": "signed"},
        {"data": "opaque_1", "type": "redacted_thinking"},
        {"type": "text", "text": "完成"},
    ]


def test_anthropic_stream_does_not_emit_unsigned_thinking_blocks():
    accumulator = AnthropicMessagesStreamAccumulator(model="glm-4")
    events = accumulator.feed_event(TextStreamEvent(kind="reasoning_delta", reasoning_content="内部推理"))
    events.extend(accumulator.feed_event(TextStreamEvent(kind="text_delta", text="完成")))
    events.extend(accumulator.feed_event(TextStreamEvent(kind="done")))

    payloads = [
        json.loads(event.split("data: ", 1)[1])
        for event in events
        if event.startswith("event: ")
    ]
    assert not any(
        payload.get("content_block", {}).get("type") == "thinking"
        for payload in payloads
        if payload["type"] == "content_block_start"
    )
    assert any(
        payload.get("content_block", {}).get("type") == "text"
        for payload in payloads
        if payload["type"] == "content_block_start"
    )


def test_anthropic_messages_to_internal_rejects_unsupported_server_tool():
    with pytest.raises(ValueError, match="服务端工具类型"):
        anthropic_messages_to_internal(
            {
                "model": "glm-4",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "computer_20250124", "name": "computer"}],
            }
        )


def test_output_adapters_accept_internal_text_response():
    result = TextGenerationResponse(
        response_id="chat_1",
        model="glm-4",
        created=1,
        message=Message(role="assistant", content="hello"),
        finish_reason="stop",
        usage=TokenUsage.estimated(input_tokens=2, output_tokens=3),
    )

    anthropic = internal_to_anthropic_messages_response(result, model="glm-4")
    responses = internal_to_openai_responses_response(result, model="glm-4")

    assert anthropic["content"] == [{"type": "text", "text": "hello"}]
    assert responses["output_text"] == "hello"
    assert responses["usage"] == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}


def test_openai_nonstream_rejects_non_internal_response():
    class WrongTypeGLM:
        def chat_completion(self, request):
            return ({"legacy": True}, None)

    handler = _build_post_test_handler(
        WrongTypeGLM(),
        {"model": "glm-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    handler.do_POST()

    assert handler.sent_statuses == [HTTPStatus.BAD_GATEWAY]
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert payload["error"]["type"] == "TypeError"
    assert "非内部文本响应" in payload["error"]["message"]


def test_http_request_body_limit_is_enforced_before_json_conversion():
    class NeverCalledGLM:
        def chat_completion(self, request):
            raise AssertionError("oversized request must be rejected before dispatch")

    handler = _build_post_test_handler(
        NeverCalledGLM(),
        {"model": "glm-4", "messages": [{"role": "user", "content": "payload"}]},
        max_request_body_bytes=1,
    )
    handler.do_POST()

    assert handler.sent_statuses == [HTTPStatus.REQUEST_ENTITY_TOO_LARGE]
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert payload["error"]["type"] == "request_too_large"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/messages", {"model": "glm-4", "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "glm-4", "input": "hi"}),
    ],
)
def test_text_http_boundaries_reject_non_internal_results(path, payload):
    class WrongTypeGLM:
        def chat_completion(self, request):
            return ({"legacy": True}, None)

    handler = _build_post_test_handler(WrongTypeGLM(), payload, path=path)
    handler.do_POST()

    assert handler.sent_statuses == [HTTPStatus.BAD_GATEWAY]
    body = json.loads(handler.wfile.getvalue().decode("utf-8"))
    if path == "/v1/messages":
        assert body["type"] == "error"
        assert body["error"]["type"] == "api_error"
    else:
        assert body["error"]["type"] == "TypeError"


def test_openai_responses_to_internal_accepts_sdk_style_input_messages():
    payload = {
        "model": "glm-4",
        "input": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                ],
            },
        ],
    }

    converted = openai_responses_to_internal(payload)

    assert converted.messages[0] == Message(role="user", content="hi")
    assert converted.messages[1] == Message(role="assistant", content="hello")
    assert converted.messages[2].role == "user"
    assert isinstance(converted.messages[2].content, tuple)
    assert converted.messages[2].content[0].kind == "text"
    assert converted.messages[2].content[1].kind == "image"
    assert converted.messages[2].content[1].url == "data:image/png;base64,abc"


def test_openai_responses_to_internal_converts_file_data_to_uploadable_data_url():
    converted = openai_responses_to_internal(
        {
            "model": "glm-5.3-flash",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_data": "AQI=",
                            "filename": "report.pdf",
                        }
                    ],
                }
            ],
        }
    )

    content = converted.messages[0].content
    assert isinstance(content, tuple)
    assert content[0].kind == "file"
    assert content[0].url == "data:application/pdf;base64,AQI="


def test_openai_responses_to_internal_rejects_unresolvable_file_id():
    with pytest.raises(ValueError, match=r"file_id.*file_url 或 file_data"):
        openai_responses_to_internal(
            {
                "model": "glm-5.3-flash",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_file", "file_id": "file_123"}],
                    }
                ],
            }
        )


def test_openai_responses_to_internal_preserves_structured_function_output():
    converted = openai_responses_to_internal(
        {
            "model": "glm-4",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "inspect",
                    "arguments": {"path": "README.md"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "文件内容"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                    ],
                },
            ],
        }
    )

    assert converted.messages[0].tool_calls[0].arguments == '{"path":"README.md"}'
    result_content = converted.messages[1].content
    assert isinstance(result_content, tuple)
    assert result_content[0].kind == "text"
    assert result_content[0].text == "文件内容"
    assert result_content[1].kind == "image"
    assert result_content[1].url == "data:image/png;base64,abc"


def test_openai_responses_to_internal_rejects_unknown_function_output_content():
    with pytest.raises(ValueError, match="暂不支持"):
        openai_responses_to_internal(
            {
                "model": "glm-4",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": [{"type": "refusal", "refusal": "not available"}],
                    }
                ],
            }
        )


def test_openai_responses_rejects_unknown_input_items_instead_of_dropping_them():
    with pytest.raises(ValueError, match="暂不支持 content 类型"):
        openai_responses_to_internal(
            {
                "model": "glm-4",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_audio", "audio_url": "x"}],
                    }
                ],
            }
        )


def test_openai_responses_rejects_malformed_input_items():
    with pytest.raises(ValueError, match=r"input\[0\] 必须是对象"):
        openai_responses_to_internal({"model": "glm-4", "input": ["invalid"]})


def test_internal_to_responses_exposes_output_text_and_standard_fields():
    response = internal_to_openai_responses_response(
        TextGenerationResponse(
            response_id="chat_1",
            model="glm-4",
            created=1,
            message=Message(role="assistant", content="hello"),
            finish_reason="stop",
            usage=TokenUsage.estimated(input_tokens=2, output_tokens=3),
        ),
        model="glm-4",
    )

    assert response["object"] == "response"
    assert response["output_text"] == "hello"
    assert response["error"] is None
    assert response["incomplete_details"] is None
    assert response["usage"] == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}


def test_responses_stream_uses_openai_event_envelope():
    accumulator = OpenAIResponsesStreamAccumulator(model="glm-4")

    events = accumulator.start_response()
    events.extend(accumulator.feed_event(TextStreamEvent(kind="text_delta", role="assistant", text="hi")))
    events.extend(accumulator.feed_event(TextStreamEvent(kind="finish", finish_reason="stop")))

    payloads = [
        json.loads(event.split("data: ", 1)[1])
        for event in events
        if event.startswith("event: ")
    ]

    assert payloads[0]["type"] == "response.created"
    assert payloads[0]["sequence_number"] == 0
    assert payloads[0]["response"]["object"] == "response"
    assert payloads[0]["response"]["usage"] is None
    assert payloads[0]["response"]["parallel_tool_calls"] is True
    text_delta = next(payload for payload in payloads if payload["type"] == "response.output_text.delta")
    assert text_delta["delta"] == "hi"
    assert text_delta["response_id"] == payloads[0]["response"]["id"]
    assert payloads[-1]["type"] == "response.completed"
    assert payloads[-1]["response"]["status"] == "completed"
    assert payloads[-1]["response"]["completed_at"] is not None
    assert payloads[-1]["response"]["usage"]["total_tokens"] == 0
    assert events[-1] == "data: [DONE]\n\n"


def test_responses_stream_finishes_after_internal_done_event():
    accumulator = OpenAIResponsesStreamAccumulator(model="glm-4")

    events = accumulator.feed_event(TextStreamEvent(kind="text_delta", text="hi"))
    events.extend(accumulator.feed_event(TextStreamEvent(kind="done")))

    payload_events = [event for event in events if event.startswith("event: ")]
    payloads = [json.loads(event.split("data: ", 1)[1]) for event in payload_events]

    assert any(payload["type"] == "response.output_text.delta" and payload["delta"] == "hi" for payload in payloads)
    assert payloads[-1]["type"] == "response.completed"
    assert events[-1] == "data: [DONE]\n\n"


def test_responses_stream_completes_on_finish_reason_without_done_sentinel():
    accumulator = OpenAIResponsesStreamAccumulator(model="glm-4")

    events = accumulator.feed_event(TextStreamEvent(kind="text_delta", text="hi"))
    events.extend(accumulator.feed_event(
        TextStreamEvent(
            kind="finish",
            finish_reason="stop",
            usage=TokenUsage.estimated(input_tokens=2, output_tokens=3),
        )
    ))

    payload_events = [event for event in events if event.startswith("event: ")]
    payloads = [json.loads(event.split("data: ", 1)[1]) for event in payload_events]

    assert payloads[-1]["type"] == "response.completed"
    assert payloads[-1]["response"]["usage"]["input_tokens"] == 2
    assert payloads[-1]["response"]["usage"]["output_tokens"] == 3
    assert events[-1] == "data: [DONE]\n\n"


def test_stream_adapters_consume_protocol_neutral_events_directly():
    text_event = TextStreamEvent(kind="text_delta", role="assistant", text="hi")
    text_finish_event = TextStreamEvent(
        kind="finish",
        finish_reason="stop",
        usage=TokenUsage.estimated(input_tokens=2, output_tokens=3),
    )
    tool_event = TextStreamEvent(
        kind="tool_call_delta",
        tool_call=ToolCallDelta(index=0, id="call_1", name="weather", arguments='{"city":"上海"}'),
    )

    anthropic = AnthropicMessagesStreamAccumulator(model="glm-4")
    anthropic_events = anthropic.feed_event(text_event)
    anthropic_events.extend(anthropic.feed_event(text_finish_event))
    anthropic_events.extend(anthropic.feed_event(TextStreamEvent(kind="done")))
    anthropic_payloads = [
        json.loads(event.split("data: ", 1)[1])
        for event in anthropic_events
        if event.startswith("event: ")
    ]
    assert any(payload["type"] == "content_block_delta" for payload in anthropic_payloads)
    assert anthropic_payloads[-2]["type"] == "message_delta"
    assert anthropic_payloads[-2]["usage"]["output_tokens"] == 3

    responses = OpenAIResponsesStreamAccumulator(model="glm-4")
    responses_events = responses.feed_event(tool_event)
    responses_events.extend(responses.feed_event(
        TextStreamEvent(
            kind="finish",
            finish_reason="tool_calls",
            usage=TokenUsage.estimated(input_tokens=2, output_tokens=3),
        )
    ))
    response_payloads = [
        json.loads(event.split("data: ", 1)[1])
        for event in responses_events
        if event.startswith("event: ")
    ]
    assert any(payload["type"] == "response.function_call_arguments.delta" for payload in response_payloads)
    completed = next(payload for payload in response_payloads if payload["type"] == "response.completed")
    assert completed["response"]["usage"]["total_tokens"] == 5


def test_anthropic_stream_serializes_tool_use_for_claude_code():
    accumulator = AnthropicMessagesStreamAccumulator(model="glm-5.3-flash")
    events = accumulator.feed_event(
        TextStreamEvent(
            kind="tool_call_delta",
            tool_call=ToolCallDelta(
                index=0,
                id="call_agent",
                name="Agent",
                arguments='{"description":"inspect","prompt":"check files"}',
            ),
        )
    )
    events.extend(
        accumulator.feed_event(
            TextStreamEvent(
                kind="finish",
                finish_reason="tool_calls",
                usage=TokenUsage.estimated(input_tokens=10, output_tokens=4),
            )
        )
    )
    events.extend(accumulator.feed_event(TextStreamEvent(kind="done")))

    payloads = [
        json.loads(event.split("data: ", 1)[1])
        for event in events
        if event.startswith("event: ")
    ]
    start = next(payload for payload in payloads if payload["type"] == "content_block_start")
    delta = next(payload for payload in payloads if payload["type"] == "content_block_delta")
    message_delta = next(payload for payload in payloads if payload["type"] == "message_delta")

    assert start["content_block"] == {"type": "tool_use", "id": "call_agent", "name": "Agent", "input": {}}
    assert delta["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"description":"inspect","prompt":"check files"}',
    }
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_responses_http_stream_sends_keepalive_while_upstream_is_idle(monkeypatch):
    monkeypatch.setattr(server_module, "RESPONSES_STREAM_HEARTBEAT_SECONDS", 0.01)

    class FakeGLM:
        def stream_chat_completion(self, payload):
            yield TextStreamEvent(kind="text_delta", role="assistant", text="hi")
            time.sleep(0.05)
            yield TextStreamEvent(kind="finish", finish_reason="stop")

    class FakeLogger:
        def debug(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass

    config = SimpleNamespace(
        host="127.0.0.1",
        port=0,
        api_prefix="/v1",
        cors_allow_origin="*",
        server_api_keys=[],
        debug_dump_all=False,
        exposed_models=["glm-4"],
    )
    server = server_module.GLM2APIServer(config, FakeGLM(), FakeLogger())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server._server.server_address[1]
    try:
        body = json.dumps({"model": "glm-4", "input": "hi", "stream": True}).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/responses",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            stream_text = response.read().decode("utf-8")
    finally:
        server.shutdown()
        thread.join(timeout=1)

    assert ": keep-alive\n\n" in stream_text
    assert "response.completed" in stream_text


def test_anthropic_messages_to_internal_maps_tool_choice_variants():
    any_payload = {
        "model": "glm-4",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "any"},
    }
    tool_payload = {
        "model": "glm-4",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }

    any_converted = anthropic_messages_to_internal(any_payload)
    tool_converted = anthropic_messages_to_internal(tool_payload)

    assert any_converted.tool_choice is not None
    assert any_converted.tool_choice.mode == "required"
    assert tool_converted.tool_choice is not None
    assert tool_converted.tool_choice.mode == "function"
    assert tool_converted.tool_choice.name == "get_weather"


def test_internal_to_anthropic_preserves_estimated_usage():
    response = internal_to_anthropic_messages_response(
        TextGenerationResponse(
            response_id="chat_1",
            model="glm-5.3-flash",
            created=1,
            message=Message(role="assistant", content="OK"),
            finish_reason="stop",
            usage=TokenUsage.estimated(input_tokens=123, output_tokens=45),
        ),
        model="glm-5.3-flash",
    )

    assert response["usage"] == {"input_tokens": 123, "output_tokens": 45}


def test_anthropic_stream_reports_input_and_output_usage():
    accumulator = AnthropicMessagesStreamAccumulator(
        model="glm-5.3-flash",
        usage=TokenUsage.estimated(input_tokens=123),
    )
    events = [accumulator.start_message()]
    events.extend(accumulator.feed_event(TextStreamEvent(kind="text_delta", text="OK")))
    events.extend(accumulator.feed_event(
        TextStreamEvent(
            kind="finish",
            finish_reason="stop",
            usage=TokenUsage.estimated(input_tokens=123, output_tokens=45),
        )
    ))
    events.extend(accumulator.feed_event(TextStreamEvent(kind="done")))

    message_start = json.loads(events[0].split("data: ", 1)[1])
    message_delta = json.loads(events[-2].split("data: ", 1)[1])

    assert message_start["message"]["usage"]["input_tokens"] == 123
    assert message_delta["usage"] == {"output_tokens": 45}


def test_glm_client_raises_for_sse_error_event():
    client = GLMWebClient.__new__(GLMWebClient)

    try:
        client.raise_for_event_error(
            {
                "status": "error",
                "last_error": {"error_code": 10025, "err_msg": "stream request error"},
                "parts": [],
            },
            stream=True,
        )
    except UpstreamAPIError as exc:
        assert exc.status_code == 502
        assert "10025" in str(exc)
        assert "stream request error" in str(exc)
    else:
        raise AssertionError("expected UpstreamAPIError")


def test_glm_client_rejects_sse_stream_without_done_marker():
    client = GLMWebClient.__new__(GLMWebClient)
    client.config = SimpleNamespace(debug_dump_all=False)

    class FakeLogger:
        def debug(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass

    client.logger = FakeLogger()

    class FakeResponse:
        def __init__(self):
            self.chunks = iter([b'data: {"status":"running"}\n\n', b""])

        def read(self, _size):
            return next(self.chunks)

    with pytest.raises(UpstreamAPIError, match="SSE 连接提前结束"):
        list(client.iter_sse_events(FakeResponse(), require_done=True))


def test_anthropic_stream_writes_error_without_message_stop_on_upstream_failure():
    class FailingGLM:
        def stream_chat_completion(self, payload):
            def generate():
                raise UpstreamAPIError(status_code=502, message="upstream disconnected")
                yield b""

            return generate()

    handler = _build_stream_test_handler(FailingGLM())
    handler._stream_anthropic(TextGenerationRequest(model="glm-4", messages=()), "glm-4")
    output = handler.wfile.getvalue().decode("utf-8")

    assert "event: error\n" in output
    assert "api_error" in output
    assert "message_stop" not in output


def test_responses_stream_writes_error_without_completion_on_upstream_failure():
    class FailingGLM:
        def stream_chat_completion(self, payload):
            def generate():
                raise UpstreamAPIError(status_code=502, message="upstream disconnected")
                yield b""

            return generate()

    handler = _build_stream_test_handler(FailingGLM())
    handler.path = "/v1/responses"
    handler._stream_responses(TextGenerationRequest(model="glm-4", messages=()), "glm-4")
    output = handler.wfile.getvalue().decode("utf-8")

    assert "event: error\n" in output
    assert '"type":"error"' in output
    assert "response.completed" not in output


def test_anthropic_error_response_uses_standard_envelope_and_request_id():
    handler = _build_stream_test_handler(object())
    handler._request_path = "/v1/messages"
    handler._request_id = "req_test"
    sent_statuses = []
    sent_headers = {}
    handler.send_response = sent_statuses.append
    handler.send_header = sent_headers.__setitem__

    handler._write_request_error(
        HTTPStatus.BAD_REQUEST,
        "请求体不是合法 JSON: syntax error",
        "invalid_json",
        legacy_payload={"error": {"message": "legacy"}},
    )

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert sent_statuses == [HTTPStatus.BAD_REQUEST]
    assert payload == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "请求体不是合法 JSON: syntax error",
        },
        "request_id": "req_test",
    }
    assert sent_headers["request-id"] == "req_test"
    assert sent_headers["Content-Type"] == "application/json; charset=utf-8"


def test_anthropic_post_invalid_json_uses_anthropic_error_envelope():
    handler = _build_stream_test_handler(object())
    handler.command = "POST"
    handler.headers = {"Content-Length": "1"}
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = io.BytesIO(b"{")
    sent_statuses = []
    sent_headers = {}
    handler.send_response = sent_statuses.append
    handler.send_header = sent_headers.__setitem__

    handler.do_POST()

    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert sent_statuses == [HTTPStatus.BAD_REQUEST]
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["request_id"] == sent_headers["request-id"]
    assert payload["request_id"].startswith("req_")


def _build_stream_test_handler(glm_client):
    class FakeLogger:
        def debug(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass

    config = SimpleNamespace(
        api_prefix="/v1",
        cors_allow_origin="*",
        debug_dump_all=False,
        server_api_keys=[],
    )
    server = server_module.GLM2APIServer.__new__(server_module.GLM2APIServer)
    server.config = config
    server.glm_client = glm_client
    server.logger = FakeLogger()
    handler_cls = server._build_handler()
    handler = object.__new__(handler_cls)
    handler.path = "/v1/messages"
    handler.wfile = io.BytesIO()
    handler.send_response = lambda *args, **kwargs: None
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda *args, **kwargs: None
    return handler


def _build_post_test_handler(glm_client, payload, *, max_request_body_bytes=None, path="/v1/chat/completions"):
    class FakeLogger:
        def debug(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass

    config = SimpleNamespace(
        api_prefix="/v1",
        cors_allow_origin="*",
        debug_dump_all=False,
        server_api_keys=[],
    )
    if max_request_body_bytes is not None:
        config.max_request_body_bytes = max_request_body_bytes
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
