import pytest

from glm2api.core.models import (
    ContentBlock,
    Message,
    TextGenerationResponse,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)
from glm2api.glm.translator import request_to_glm_payload
from glm2api.api.adapters.openai.chat_completions import (
    internal_to_openai_chat_completions_response,
    openai_chat_completions_to_internal,
)
from glm2api.media.images import openai_images_to_internal
from glm2api.core.usage import TokenUsage


def test_chat_file_block_converts_nested_file_data_to_data_url():
    request = openai_chat_completions_to_internal(
        {
            "model": "glm-5.3-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {"file_data": "AQI=", "filename": "report.pdf"},
                        }
                    ],
                }
            ],
        }
    )

    content = request.messages[0].content
    assert isinstance(content, tuple)
    assert content[0].kind == "file"
    assert content[0].url == "data:application/pdf;base64,AQI="


def test_chat_file_block_rejects_unresolvable_file_id():
    with pytest.raises(ValueError, match=r"file_id.*file_url 或 file_data"):
        openai_chat_completions_to_internal(
            {
                "model": "glm-5.3-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "file", "file": {"file_id": "file_123"}}],
                    }
                ],
            }
        )


def test_chat_adapter_rejects_unknown_content_instead_of_dropping_it():
    with pytest.raises(ValueError, match="暂不支持 content block 类型"):
        openai_chat_completions_to_internal(
            {
                "model": "glm-5.3-flash",
                "messages": [
                    {"role": "user", "content": [{"type": "input_audio", "audio_url": "x"}]},
                ],
            }
        )


def test_chat_adapter_rejects_malformed_message_list():
    with pytest.raises(ValueError, match="messages 必须是数组"):
        openai_chat_completions_to_internal({"model": "glm-4", "messages": "not-a-list"})


def test_chat_adapter_accepts_new_max_completion_tokens_alias():
    request = openai_chat_completions_to_internal(
        {"model": "glm-4", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 64}
    )
    assert request.max_tokens == 64


def test_thinking_blocks_round_trip_signature_and_redacted_data():
    payload = {
        "model": "glm-4",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "先分析", "signature": "sig_1"},
                    {"type": "redacted_thinking", "data": "opaque_1"},
                ],
            }
        ],
        "stream": False,
    }

    request = openai_chat_completions_to_internal(payload)

    assert request_to_glm_payload(request) == payload


def test_text_request_round_trips_without_openai_objects_inside():
    payload = {
        "model": "glm-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "查天气"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/map.png", "detail": "low"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"上海"}'},
                    }
                ],
            },
        ],
        "stream": True,
        "max_tokens": 128,
        "temperature": 0,
        "top_p": 0.9,
        "stop": ["END"],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "查询天气",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "weather"}},
        "reasoning_effort": "high",
    }

    request = openai_chat_completions_to_internal(payload)

    assert isinstance(request.messages[0], Message)
    assert isinstance(request.messages[0].content, tuple)
    assert isinstance(request.messages[0].content[0], ContentBlock)
    assert isinstance(request.messages[1].tool_calls[0], ToolCall)
    assert isinstance(request.tools[0], ToolDefinition)
    assert request.tool_choice == ToolChoice(mode="function", name="weather")
    assert request_to_glm_payload(request) == payload


def test_image_request_is_independent_from_text_request():
    request = openai_images_to_internal(
        {
            "model": "glm-image-1",
            "prompt": "一只猫",
            "size": "1024x1024",
            "n": 10,
            "response_format": "b64_json",
            "style": "none",
            "scene": "none",
        },
        default_model="glm-image-1",
    )

    assert request.n == 10
    assert request.prompt == "一只猫"
    assert request.response_format == "b64_json"


def test_text_response_exposes_semantic_fields_and_legacy_payload():
    message = Message(
        role="assistant",
        content="完成",
        tool_calls=(ToolCall(id="call_1", name="save", arguments='{"ok":true}'),),
    )
    usage = TokenUsage.estimated(input_tokens=10, output_tokens=4)

    response = TextGenerationResponse(
        response_id="chat_1",
        model="glm-4",
        created=1,
        message=message,
        finish_reason="tool_calls",
        usage=usage,
    )

    assert not isinstance(response, dict)
    assert response.message is message
    assert response.usage is usage
    payload = internal_to_openai_chat_completions_response(response)
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "save"
