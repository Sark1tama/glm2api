import json

import pytest

from glm2api.glm.events import GLMUpstreamEventAccumulator
from glm2api.glm.tools.dsml import BLOCKED_NATIVE_TOOL_NAMES
from glm2api.glm.translator import (
    convert_messages_to_glm_prompt,
    extract_text_content,
    sanitize_tool_call_payload,
)
from glm2api.core.usage import TokenUsage, estimate_conservative_prompt_tokens, estimate_conservative_tokens
from glm2api.api.adapters.openai.chat_completions import (
    internal_to_openai_chat_completions_response,
    serialize_openai_chat_completions_stream_event,
)
from glm2api.core.models import Message, ToolCall, ToolChoice, ToolDefinition


def _openai_payload(response):
    """Serialize an internal response only when asserting the HTTP view."""
    return internal_to_openai_chat_completions_response(response)


def _openai_chunks(events):
    """Serialize internal stream events only at the OpenAI boundary."""
    return [chunk for event in events for chunk in serialize_openai_chat_completions_stream_event(event)]


def _tool(name, description="", parameters=None):
    return ToolDefinition(name=name, description=description, parameters=parameters or {})


def test_conservative_token_estimate_is_deliberately_high_for_mixed_text():
    text = "你好，world!"

    assert estimate_conservative_tokens(text) > len(text)
    assert estimate_conservative_prompt_tokens(text) > estimate_conservative_tokens(text)


def test_accumulator_uses_estimated_usage_instead_of_placeholder_values():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", input_tokens_estimate=123)
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [{"type": "text", "text": "你好，世界"}],
                }
            ],
        }
    )

    usage = _openai_payload(accumulator.build_response())["usage"]

    assert usage["prompt_tokens"] == 123
    assert usage["completion_tokens"] > 1
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert accumulator.usage is not None
    assert accumulator.usage.source == "estimated"


def test_accumulator_preserves_input_estimate_from_usage_object():
    accumulator = GLMUpstreamEventAccumulator(
        model="glm-test",
        usage=TokenUsage.estimated(input_tokens=123),
    )
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [{"type": "text", "text": "你好，世界"}],
                }
            ],
        }
    )

    usage = _openai_payload(accumulator.build_response())["usage"]

    assert usage["prompt_tokens"] == 123
    assert usage["completion_tokens"] > 1


def test_accumulator_handles_delta_events_then_finish_snapshot():
    answer = "检查 Python 项目依赖版本。"
    accumulator = GLMUpstreamEventAccumulator(model="glm-test")
    payloads = [
        {"status": "init", "parts": [{"logic_id": "1", "content": [{"type": "text", "text": "检查"}]}]},
        {"status": "init", "parts": [{"logic_id": "1", "content": [{"type": "text", "text": " Python"}]}]},
        {"status": "init", "parts": [{"logic_id": "1", "content": [{"type": "text", "text": " 项目"}]}]},
        {"status": "init", "parts": [{"logic_id": "1", "content": [{"type": "text", "text": "依赖"}]}]},
        {"status": "finish", "parts": [{"logic_id": "1", "content": [{"type": "text", "text": answer}]}]},
    ]

    events: list = []
    for payload in payloads:
        stream_events, _ = accumulator.consume_event(payload)
        events.extend(stream_events)
    events.extend(accumulator.finalize(status="finish"))

    streamed_text = "".join(event.text for event in events if event.kind == "text_delta")
    assert streamed_text == answer
    assert _openai_payload(accumulator.build_response())["choices"][0]["message"]["content"] == answer


def test_accumulator_prefers_usage_reported_by_upstream_when_available():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", input_tokens_estimate=123)
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "usage": {"prompt_tokens": 7, "completion_tokens": 5},
            "parts": [
                {
                    "logic_id": "1",
                    "content": [{"type": "text", "text": "OK"}],
                }
            ],
        }
    )

    response = accumulator.build_response()

    assert _openai_payload(response)["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
    }
    assert accumulator.usage is not None
    assert accumulator.usage.source == "upstream"


def test_accumulator_stream_finalize_includes_estimated_usage():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", input_tokens_estimate=80)
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [{"type": "text", "text": "流式回答"}],
                }
            ],
        }
    )

    final_chunks = _openai_chunks(accumulator.finalize(status="finish"))
    usage_payload = json.loads(final_chunks[-2].split("data: ", 1)[1])

    assert usage_payload["usage"]["prompt_tokens"] == 80
    assert usage_payload["usage"]["completion_tokens"] > 1


def test_accumulator_counts_intervention_text_in_estimated_output_usage():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", input_tokens_estimate=80)
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [{"type": "text", "text": "原始回答"}],
                }
            ],
        }
    )

    baseline = estimate_conservative_tokens("原始回答")
    final_chunks = _openai_chunks(
        accumulator.finalize(
            status="intervene",
            last_error={"intervene_text": "这是一段需要返回给客户端的干预说明。"},
        )
    )
    usage_payload = json.loads(final_chunks[-2].split("data: ", 1)[1])

    assert usage_payload["usage"]["completion_tokens"] > baseline


def test_convert_messages_to_glm_prompt_injects_xml_tool_prompt_and_history():
    converted = convert_messages_to_glm_prompt(
        messages=[
            Message(role="user", content="查天气"),
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(id="call_1", name="get_weather", arguments='{"city":"上海"}'),
                ),
            ),
            Message(role="tool", name="get_weather", tool_call_id="call_1", content="晴"),
        ],
        tools=[
            _tool(
                "get_weather",
                "查询天气",
                {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ],
    )

    prompt = converted[0]["content"][0]["text"]

    assert "<|DSML|tool_calls>" in prompt
    assert "<|DSML|invoke name=\"get_weather\">" in prompt
    assert "<|DSML|tool_result call_id=\"call_1\" name=\"get_weather\">" in prompt
    assert "<ml_tool_calls>" not in prompt
    assert "# TOOL USE PROTOCOL" in prompt
    assert "Use the DSML format below exactly." in prompt
    assert "The server will parse this DSML block back into standard OpenAI tool_calls." in prompt
    assert "<|DSML|parameter name=\"actual_parameter_name\"><![CDATA[value]]></|DSML|parameter>" in prompt
    assert "Each argument must be a <|DSML|parameter name=\"...\"> child of the invoke." in prompt
    assert "Parameter names are case-sensitive and must exactly match the schema." in prompt
    assert "never change it to `filepath`, `file_path`, or `FilePath`." in prompt
    assert "# BLOCKED TOOLS" not in prompt
    assert "Ignore any tool names that are not listed below" in prompt


def test_convert_messages_to_glm_prompt_rejects_orphan_and_duplicate_tool_results():
    with pytest.raises(ValueError, match="缺少"):
        convert_messages_to_glm_prompt(
            messages=[Message(role="tool", content="结果")],
            tools=[],
        )

    with pytest.raises(ValueError, match="找不到对应"):
        convert_messages_to_glm_prompt(
            messages=[Message(role="tool", tool_call_id="missing", content="结果")],
            tools=[],
        )

    messages = [
        Message(
            role="assistant",
            tool_calls=(ToolCall(id="call_1", name="get_weather", arguments="{}"),),
        ),
        Message(role="tool", tool_call_id="call_1", content="第一次"),
        Message(role="tool", tool_call_id="call_1", content="第二次"),
    ]
    with pytest.raises(ValueError, match="重复"):
        convert_messages_to_glm_prompt(messages=messages, tools=[_tool("get_weather")])


def test_accumulator_build_response_maps_xml_to_openai_tool_calls():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", allowed_tool_names={"get_weather"})
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "<|DSML|tool_calls><|DSML|invoke name=\"get_weather\">"
                            "<|DSML|parameter name=\"city\">上海</|DSML|parameter>"
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    payload = _openai_payload(response)
    message = payload["choices"][0]["message"]

    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"city":"上海"}'


def test_accumulator_streaming_tool_call_emits_assistant_role_before_tool_delta():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", allowed_tool_names={"write"})
    events, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "<|DSML|tool_calls><|DSML|invoke name=\"write\">"
                            "<|DSML|parameter name=\"filePath\">test.txt</|DSML|parameter>"
                            "<|DSML|parameter name=\"content\"></|DSML|parameter>"
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    chunks = _openai_chunks(events)
    final_chunks = _openai_chunks(accumulator.finalize(status))

    assert chunks == []
    assert '"delta":{"role":"assistant"}' in final_chunks[0]
    assert '"tool_calls"' in final_chunks[1]
    assert '"finish_reason":"tool_calls"' in final_chunks[2]


def test_accumulator_streaming_extracts_tool_call_from_reasoning_fallback():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", allowed_tool_names={"write"})
    events, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "think",
                            "think": "I should call the tool.\n"
                            "<|DSML|tool_calls><|DSML|invoke name=\"write\">"
                            "<|DSML|parameter name=\"filePath\">test.txt</|DSML|parameter>"
                            "<|DSML|parameter name=\"content\"></|DSML|parameter>"
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    chunks = _openai_chunks(events)
    final_chunks = _openai_chunks(accumulator.finalize(status))

    assert chunks
    assert '"reasoning_content"' in chunks[0]
    assert '"delta":{"role":"assistant"}' in final_chunks[0]
    assert '"tool_calls"' in final_chunks[1]
    assert '\\"filePath\\":\\"test.txt\\"' in final_chunks[1]


def test_accumulator_build_response_extracts_tool_call_from_reasoning_fallback():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", allowed_tool_names={"write"})
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "think",
                            "think": "<|DSML|tool_calls><|DSML|invoke name=\"write\">"
                            "<|DSML|parameter name=\"filePath\">test.txt</|DSML|parameter>"
                            "<|DSML|parameter name=\"content\"></|DSML|parameter>"
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    payload = _openai_payload(response)
    message = payload["choices"][0]["message"]

    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "write"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"filePath":"test.txt","content":""}'


def test_sanitize_shell_command_argument_from_json_string():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": '["powershell.exe","-Command","Get-ChildItem -Force"]',
            "workdir": "E:\\Projects\\2api\\glm2api",
        },
    )

    assert cleaned == {
        "command": ["powershell.exe", "-Command", "Get-ChildItem -Force"],
        "workdir": "E:\\Projects\\2api\\glm2api",
    }


def test_sanitize_shell_command_argument_from_quoted_sequence():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": '"powershell.exe", "-Command", "Get-ChildItem -Force"',
        },
    )

    assert cleaned == {
        "command": ["powershell.exe", "-Command", "Get-ChildItem -Force"],
    }


def test_sanitize_shell_command_argument_from_plain_string():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": "Get-ChildItem",
        },
    )

    assert cleaned == {
        "command": ["powershell.exe", "-Command", "Get-ChildItem"],
    }


def test_sanitize_shell_command_argument_wraps_powershell_cmdlet_array():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": ["Get-ChildItem", "-Recurse", "-Filter", "*.txt"],
        },
    )

    assert cleaned == {
        "command": ["powershell.exe", "-Command", "Get-ChildItem -Recurse -Filter *.txt"],
    }


def test_sanitize_shell_command_argument_keeps_native_executable_array():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": ["git", "status", "--short"],
        },
    )

    assert cleaned == {
        "command": ["git", "status", "--short"],
    }


def test_accumulator_drops_tool_preamble_and_repairs_shell_command_array():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", allowed_tool_names={"shell"})
    events, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "我将创建文件。\n\n"
                            '<|DSML|tool_calls><|DSML|invoke name="shell">'
                            '<|DSML|parameter name="command"><![CDATA[["powershell.exe", "-Command", "pwd"]]></|DSML|parameter>'
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    chunks = _openai_chunks(events)
    final_chunks = _openai_chunks(accumulator.finalize(status))

    assert chunks == []
    assert "我将创建文件" not in "".join(final_chunks)
    assert '"tool_calls"' in final_chunks[1]
    assert '\\"command\\":[\\"powershell.exe\\",\\"-Command\\",\\"pwd\\"]' in final_chunks[1]


def test_accumulator_defers_visible_text_when_tools_available():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", allowed_tool_names={"shell"})
    events, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [{"type": "text", "text": "你好"}],
                }
            ],
        }
    )

    chunks = _openai_chunks(events)
    final_chunks = _openai_chunks(accumulator.finalize(status))

    assert chunks == []
    assert '"content":"你好"' in final_chunks[0]
    assert '"finish_reason":"stop"' in final_chunks[1]


def test_accumulator_reports_unavailable_dsml_tool_instead_of_empty_response():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", allowed_tool_names={"shell"})
    events, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": '<|DSML|tool_calls><|DSML|invoke name="search">'
                            '<|DSML|parameter name="search_query"><![CDATA[{"q":"阿房宫赋","recency":365}]></|DSML|parameter>'
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    chunks = _openai_chunks(events)
    final_chunks = _openai_chunks(accumulator.finalize(status))

    assert chunks == []
    assert "未声明工具" in final_chunks[0]
    assert "`search`" in final_chunks[0]
    assert '"finish_reason":"stop"' in final_chunks[1]


def test_convert_messages_to_glm_prompt_respects_tool_choice_none_and_specific():
    none_converted = convert_messages_to_glm_prompt(
        messages=[Message(role="user", content="直接回答")],
        tools=[_tool("get_weather", "查询天气", {"type": "object"})],
        tool_choice=ToolChoice(mode="none"),
    )
    none_prompt = none_converted[0]["content"][0]["text"]
    assert "# TOOL SCHEMAS" not in none_prompt

    specific_converted = convert_messages_to_glm_prompt(
        messages=[Message(role="user", content="查天气")],
        tools=[_tool("get_weather", "查询天气", {"type": "object"})],
        tool_choice=ToolChoice(mode="function", name="get_weather"),
    )
    specific_prompt = specific_converted[0]["content"][0]["text"]
    assert "You must call exactly `get_weather` before giving a final answer." in specific_prompt


def test_accumulator_enforces_tool_choice_for_text_and_streaming_results():
    none_accumulator = GLMUpstreamEventAccumulator(
        model="glm-test",
        allowed_tool_names=set(),
        tool_choice=ToolChoice(mode="none"),
    )
    none_accumulator.consume_event(
        {
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": '<|DSML|tool_calls><|DSML|invoke name="get_weather">'
                            '<|DSML|parameter name="city">上海</|DSML|parameter>'
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="none"):
        none_accumulator.build_response()

    required_accumulator = GLMUpstreamEventAccumulator(
        model="glm-test",
        allowed_tool_names={"get_weather"},
        tool_choice=ToolChoice(mode="required"),
    )
    required_accumulator.consume_event(
        {
            "parts": [{"logic_id": "1", "content": [{"type": "text", "text": "普通回答"}]}]
        }
    )
    with pytest.raises(ValueError, match="未输出工具调用"):
        required_accumulator.finalize(status="finish")


def test_convert_messages_to_glm_prompt_filters_native_url_tools_and_reinforces_tool_awareness():
    converted = convert_messages_to_glm_prompt(
        messages=[Message(role="user", content="打开 https://example.com")],
        tools=[
            _tool("open_url", "Open URL", {"type": "object"}),
            _tool("mcp__CherryFetch__fetchJson", "Fetch JSON", {"type": "object"}),
        ],
        blocked_tool_names=BLOCKED_NATIVE_TOOL_NAMES,
    )

    prompt = converted[0]["content"][0]["text"]

    assert "Tool: open_url" not in prompt
    assert "Server-side native tools" not in prompt
    assert "Tool: mcp__CherryFetch__fetchJson" in prompt
    assert "You do not have hidden browser, web, or URL-opening tools." in prompt
    assert "Never call native tools such as `open_url`" in prompt
    assert "Do not output hidden reasoning, chain-of-thought, or labels such as `Thinking:`." in prompt
    assert "Do not narrate tool selection, failed tool attempts, retries, fallback plans, or tool status banners." in prompt
    assert "Never output tool-call display text such as `⚙ tool_name [...]`" in prompt


def test_convert_messages_to_glm_prompt_drops_blocked_tool_call_history():
    converted = convert_messages_to_glm_prompt(
        messages=[
            Message(role="user", content="打开 https://example.com"),
            Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="call_bad",
                        name="open_url",
                        arguments='{"url":"https://example.com"}',
                    ),
                ),
            ),
        ],
        tools=[_tool("mcp__CherryFetch__fetchJson", "Fetch JSON", {"type": "object"})],
    )

    prompt = converted[0]["content"][0]["text"]

    assert "name=\"open_url\"" not in prompt
    assert "Tool: mcp__CherryFetch__fetchJson" in prompt


def test_convert_messages_to_glm_prompt_repairs_cherry_fetch_url_and_skips_invalid_tool_error_history():
    converted = convert_messages_to_glm_prompt(
        messages=[
            Message(
                role="user",
                content="使用工具访问 https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8",
            ),
            Message(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCall(
                        id="call_bad",
                        name="mcp__CherryFetch__fetchJson",
                        arguments='{"param_name":"url"}',
                    ),
                ),
            ),
            Message(
                role="tool",
                tool_call_id="call_bad",
                content="{\"isError\":true,\"content\":[{\"type\":\"text\",\"text\":\"Invalid input: expected string, received undefined\"}]}",
            ),
        ],
        tools=[
            _tool(
                "mcp__CherryFetch__fetchJson",
                "Fetch a JSON file from a URL",
                {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            )
        ],
    )

    prompt = converted[0]["content"][0]["text"]

    assert (
        "<|DSML|parameter name=\"url\"><![CDATA[https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8]]></|DSML|parameter>"
        in prompt
    )
    assert "expected string, received undefined" not in prompt


def test_accumulator_repairs_param_name_only_tool_call_with_fallback_url():
    accumulator = GLMUpstreamEventAccumulator(
        model="glm-test",
        allowed_tool_names={"mcp__CherryFetch__fetchJson"},
        fallback_tool_url="https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8",
    )
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "<ml_tool_calls><ml_tool_call><ml_tool_name>mcp__CherryFetch__fetchJson</ml_tool_name>"
                            "<ml_parameters><param_name><![CDATA[url]]></param_name></ml_parameters>"
                            "</ml_tool_call></ml_tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    payload = _openai_payload(response)
    message = payload["choices"][0]["message"]

    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "mcp__CherryFetch__fetchJson"
    assert (
        message["tool_calls"][0]["function"]["arguments"]
        == '{"url":"https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8"}'
    )


def test_accumulator_ignores_unallowed_native_tool_call_blocks():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test", allowed_tool_names={"get_weather"})
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "tool_calls",
                            "tool_calls": {
                                "id": "call_open_url",
                                "name": "open_url",
                                "arguments": '{"url":"https://example.com"}',
                            },
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    payload = _openai_payload(response)
    message = payload["choices"][0]["message"]

    assert payload["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in message


def test_accumulator_does_not_expose_native_tools_without_declarations():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test")
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "tool_calls",
                            "tool_calls": {
                                "id": "call_finish",
                                "name": "finish",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            ],
        }
    )

    message = _openai_payload(accumulator.build_response())["choices"][0]["message"]

    assert "tool_calls" not in message


def test_accumulator_keeps_markdown_block_separators_between_parts():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test")

    first_events, _ = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {"type": "text", "text": "## 查询结果：IP 地址 `1.1.1.1` 的归属地信息"},
                    ],
                }
            ],
        }
    )
    second_events, _ = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "2",
                    "content": [
                        {"type": "text", "text": "| 字段 | 值 |\n|---|---|\n| 查询 IP | `1.1.1.1` |"},
                    ],
                }
            ],
        }
    )

    first_chunks = _openai_chunks(first_events)
    second_chunks = _openai_chunks(second_events)
    assert first_chunks
    assert second_chunks[0].find("\\n\\n") != -1

    response = accumulator.build_response()
    assert _openai_payload(response)["choices"][0]["message"]["content"] == (
        "## 查询结果：IP 地址 `1.1.1.1` 的归属地信息\n\n"
        "| 字段 | 值 |\n|---|---|\n| 查询 IP | `1.1.1.1` |"
    )


def test_accumulator_preserves_arrival_order_for_opaque_logic_ids():
    accumulator = GLMUpstreamEventAccumulator(model="glm-test")
    accumulator.consume_event(
        {
            "parts": [{"logic_id": "z-part", "content": [{"type": "text", "text": "first"}]}],
        }
    )
    accumulator.consume_event(
        {
            "parts": [{"logic_id": "a-part", "content": [{"type": "text", "text": "second"}]}],
        }
    )

    response = accumulator.build_response()
    assert _openai_payload(response)["choices"][0]["message"]["content"] == "first\n\nsecond"
