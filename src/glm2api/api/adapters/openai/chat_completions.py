"""OpenAI Chat Completions request and response conversion."""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from ....core.models import (
    ContentBlock,
    Message,
    TextGenerationRequest,
    TextGenerationResponse,
    TextStreamEvent,
    ToolCall,
    ToolDefinition,
)
from ....core.usage import TokenUsage
from ....utils.json import safe_json_dumps
from .common import file_data_to_data_url, tool_choice_from_openai


_OPENAI_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool", "developer"})


def _content_block_from_openai(value: object, path: str = "content") -> ContentBlock:
    if not isinstance(value, dict):
        raise ValueError(f"OpenAI {path} 必须是对象")
    block_type = value.get("type")
    if not isinstance(block_type, str) or not block_type.strip():
        raise ValueError(f"OpenAI {path}.type 缺失或无效")
    block_type = block_type.strip()
    if block_type == "text":
        text = value.get("text")
        if not isinstance(text, str):
            raise ValueError(f"OpenAI {path}.text 必须是字符串")
        return ContentBlock(kind="text", text=text)
    if block_type == "thinking":
        thinking = value.get("thinking")
        if not isinstance(thinking, str):
            raise ValueError(f"OpenAI {path}.thinking 必须是字符串")
        metadata = dict(value)
        metadata.pop("type", None)
        metadata.pop("thinking", None)
        return ContentBlock(
            kind="thinking",
            text=thinking,
            metadata=metadata,
        )
    if block_type == "redacted_thinking":
        if not isinstance(value.get("data"), str) or not value["data"]:
            raise ValueError(f"OpenAI {path}.data 不能为空")
        metadata = dict(value)
        metadata.pop("type", None)
        return ContentBlock(kind="redacted_thinking", metadata=metadata)
    if block_type == "image_url":
        raw_url = value.get("image_url")
        if isinstance(raw_url, dict):
            url = raw_url.get("url")
            detail = raw_url.get("detail")
        else:
            url = raw_url
            detail = value.get("detail")
        if isinstance(url, str) and url.strip():
            return ContentBlock(kind="image", url=url, detail=str(detail) if detail else None)
        raise ValueError(f"OpenAI {path}.image_url.url 不能为空")
    if block_type == "file":
        raw_file = value.get("file_url")
        url = raw_file.get("url") if isinstance(raw_file, dict) else raw_file
        nested_file = value.get("file")
        nested_file = nested_file if isinstance(nested_file, dict) else {}
        if not (isinstance(url, str) and url.strip()):
            nested_url = nested_file.get("file_url") or nested_file.get("url")
            url = nested_url.get("url") if isinstance(nested_url, dict) else nested_url
        if isinstance(url, str) and url.strip():
            return ContentBlock(kind="file", url=url.strip())

        file_data = value.get("file_data")
        if file_data is None:
            file_data = nested_file.get("file_data")
        filename = value.get("filename") or nested_file.get("filename")
        data_url = file_data_to_data_url(file_data, filename)
        if data_url:
            return ContentBlock(kind="file", url=data_url)

        file_id = value.get("file_id") or nested_file.get("file_id")
        if file_id is not None and str(file_id).strip():
            raise ValueError("OpenAI file_id 无法由本地代理解析，请改用 file_url 或 file_data")
        raise ValueError("OpenAI file block 必须包含 file_url 或 file_data")
    raise ValueError(f"OpenAI {path} 暂不支持 content block 类型: {block_type}")


def _tool_call_from_openai(value: object, path: str = "tool_calls") -> ToolCall:
    if not isinstance(value, dict):
        raise ValueError(f"OpenAI {path} 必须是对象")
    function = value.get("function")
    if not isinstance(function, dict):
        raise ValueError(f"OpenAI {path}.function 必须是对象")
    name = str(function.get("name", "")).strip()
    if not name:
        raise ValueError(f"OpenAI {path}.function.name 不能为空")
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        raise ValueError(f"OpenAI {path}.function.arguments 必须是 JSON 字符串")
    return ToolCall(
        id=str(value.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
        name=name,
        arguments=arguments,
    )


def _tool_definition_from_openai(value: object, path: str = "tools") -> ToolDefinition:
    if not isinstance(value, dict):
        raise ValueError(f"OpenAI {path} 必须是对象")
    if value.get("type") != "function":
        raise ValueError(f"OpenAI {path}.type 只支持 function")
    function = value.get("function")
    if not isinstance(function, dict):
        raise ValueError(f"OpenAI {path}.function 必须是对象")
    name = str(function.get("name", "")).strip()
    if not name:
        raise ValueError(f"OpenAI {path}.function.name 不能为空")
    parameters = function.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"OpenAI {path}.function.parameters 必须是对象")
    strict = function.get("strict")
    return ToolDefinition(
        name=name,
        description=str(function.get("description", "") or ""),
        parameters=dict(parameters),
        strict=strict if isinstance(strict, bool) else None,
    )


def _message_from_openai(value: object, path: str = "messages") -> Message:
    if not isinstance(value, dict):
        raise ValueError(f"OpenAI {path} 必须是对象")
    role = value.get("role")
    if not isinstance(role, str) or role not in _OPENAI_MESSAGE_ROLES:
        raise ValueError(f"OpenAI {path}.role 不受支持")
    raw_content = value.get("content")
    if isinstance(raw_content, list):
        blocks = tuple(
            _content_block_from_openai(raw_block, f"{path}.content[{index}]")
            for index, raw_block in enumerate(raw_content)
        )
        content = blocks
    elif isinstance(raw_content, str) or raw_content is None:
        content = raw_content
    else:
        raise ValueError(f"OpenAI {path}.content 必须是字符串、null 或 block 数组")

    raw_tool_calls = value.get("tool_calls")
    if raw_tool_calls is not None and not isinstance(raw_tool_calls, list):
        raise ValueError(f"OpenAI {path}.tool_calls 必须是数组")
    raw_tool_call_list = raw_tool_calls if isinstance(raw_tool_calls, list) else []
    tool_calls = tuple(
        _tool_call_from_openai(raw_tool_call, f"{path}.tool_calls[{index}]")
        for index, raw_tool_call in enumerate(raw_tool_call_list)
    )
    tool_call_id = value.get("tool_call_id")
    if tool_call_id is not None and (not isinstance(tool_call_id, str) or not tool_call_id.strip()):
        raise ValueError(f"OpenAI {path}.tool_call_id 必须是非空字符串")
    name = value.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"OpenAI {path}.name 必须是非空字符串")
    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id.strip() if isinstance(tool_call_id, str) else None,
        name=name.strip() if isinstance(name, str) else None,
        reasoning_content=(
            str(value["reasoning_content"])
            if value.get("reasoning_content") is not None
            else None
        ),
    )


def _content_block_to_openai(block: ContentBlock) -> dict[str, object]:
    if block.kind == "text":
        return {"type": "text", "text": block.text or ""}
    if block.kind == "image":
        image_url: dict[str, object] = {"url": block.url or ""}
        if block.detail:
            image_url["detail"] = block.detail
        return {"type": "image_url", "image_url": image_url}
    if block.kind == "file":
        return {"type": "file", "file_url": {"url": block.url or ""}}
    if block.kind == "thinking":
        result = dict(block.metadata)
        result["type"] = "thinking"
        result["thinking"] = block.text or ""
        return result
    if block.kind == "redacted_thinking":
        result = dict(block.metadata)
        result["type"] = "redacted_thinking"
        return result
    return dict(block.metadata)


def _tool_call_to_openai(tool_call: ToolCall) -> dict[str, object]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
        },
    }


def _message_to_openai(message: Message) -> dict[str, object]:
    if isinstance(message.content, tuple):
        if len(message.content) == 1 and message.content[0].kind == "text":
            content: object = message.content[0].text or ""
        else:
            content = [_content_block_to_openai(block) for block in message.content]
    else:
        content = message.content

    result: dict[str, object] = {"role": message.role, "content": content}
    if message.name is not None:
        result["name"] = message.name
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        result["tool_calls"] = [_tool_call_to_openai(tool_call) for tool_call in message.tool_calls]
    if message.reasoning_content is not None:
        result["reasoning_content"] = message.reasoning_content
    return result


def serialize_openai_usage(usage: TokenUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def openai_chat_completions_to_internal(payload: Mapping[str, object]) -> TextGenerationRequest:
    """Convert an OpenAI Chat Completions request into the internal text model."""
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("OpenAI messages 必须是数组")
    messages = tuple(
        _message_from_openai(raw_message, f"messages[{index}]")
        for index, raw_message in enumerate(raw_messages)
    )

    raw_tools = payload.get("tools")
    if raw_tools is not None and not isinstance(raw_tools, list):
        raise ValueError("OpenAI tools 必须是数组")
    raw_tool_list = raw_tools if isinstance(raw_tools, list) else []
    tools = tuple(
        _tool_definition_from_openai(raw_tool, f"tools[{index}]")
        for index, raw_tool in enumerate(raw_tool_list)
    )

    raw_stop = payload.get("stop")
    if isinstance(raw_stop, list):
        stop: str | tuple[str, ...] | None = tuple(str(item) for item in raw_stop)
    elif isinstance(raw_stop, str):
        stop = raw_stop
    elif raw_stop is None:
        stop = None
    else:
        raise ValueError("OpenAI stop 必须是字符串、字符串数组或 null")
    if isinstance(raw_stop, list) and any(not isinstance(item, str) for item in raw_stop):
        raise ValueError("OpenAI stop 数组元素必须是字符串")

    response_format = payload.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        raise ValueError("OpenAI response_format 必须是对象")
    if response_format is None:
        response_format = None

    for field_name in ("stream", "web_search", "deep_research"):
        if field_name in payload and not isinstance(payload[field_name], bool):
            raise ValueError(f"OpenAI {field_name} 必须是布尔值")
    max_tokens = payload.get("max_tokens")
    max_completion_tokens = payload.get("max_completion_tokens")
    if max_tokens is not None and max_completion_tokens is not None:
        raise ValueError("OpenAI max_tokens 与 max_completion_tokens 不能同时提供")
    if max_tokens is None:
        max_tokens = max_completion_tokens
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0
    ):
        raise ValueError("OpenAI max_tokens 必须是非负整数")
    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    for field_name, value in (("temperature", temperature), ("top_p", top_p)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"OpenAI {field_name} 必须是数字")
    reasoning_effort = payload.get("reasoning_effort")
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        raise ValueError("OpenAI reasoning_effort 必须是字符串")

    known_fields = {
        "model",
        "messages",
        "stream",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
        "reasoning_effort",
        "web_search",
        "deep_research",
    }
    extra = {key: value for key, value in payload.items() if key not in known_fields}
    return TextGenerationRequest(
        model=str(payload.get("model") or ""),
        messages=messages,
        stream=bool(payload.get("stream", False)),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice_from_openai(payload.get("tool_choice")),
        response_format=dict(response_format) if response_format is not None else None,
        reasoning_effort=reasoning_effort,
        web_search=bool(payload.get("web_search", False)),
        deep_research=bool(payload.get("deep_research", False)),
        extra=extra,
    )


def internal_to_openai_chat_completions_response(result: TextGenerationResponse) -> dict[str, object]:
    """Convert an internal text result to an OpenAI Chat Completions response."""
    return {
        "id": result.response_id,
        "object": "chat.completion",
        "created": result.created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": _message_to_openai(result.message),
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": serialize_openai_usage(result.usage),
    }


def serialize_openai_chat_completions_stream_event(event: TextStreamEvent) -> list[str]:
    """Serialize one internal event into legacy OpenAI SSE blocks."""
    if event.kind == "done":
        return ["data: [DONE]\n\n"]

    if event.kind == "text_delta":
        delta: dict[str, object] = {"content": event.text}
        if event.role:
            delta["role"] = event.role
        return [_chunk(event, {"choices": [_choice(delta)]})]

    if event.kind == "reasoning_delta":
        return [_chunk(event, {"choices": [_choice({"reasoning_content": event.reasoning_content})]})]

    if event.kind == "role":
        return [_chunk(event, {"choices": [_choice({"role": event.role or "assistant"})]})]

    if event.kind == "tool_call_delta" and event.tool_call is not None:
        tool_call = event.tool_call
        function: dict[str, object] = {}
        if tool_call.name is not None:
            function["name"] = tool_call.name
        if tool_call.arguments:
            function["arguments"] = tool_call.arguments
        serialized_call: dict[str, object] = {
            "index": tool_call.index,
            "function": function,
        }
        if tool_call.id is not None:
            serialized_call.update({"id": tool_call.id, "type": "function"})
        return [_chunk(event, {"choices": [_choice({"tool_calls": [serialized_call]})]})]

    if event.kind == "finish":
        patch: dict[str, object] = {
            "choices": [_choice({}, finish_reason=event.finish_reason)],
        }
        if event.usage is not None:
            patch["usage"] = serialize_openai_usage(event.usage)
        return [_chunk(event, patch)]

    return []


def _choice(delta: dict[str, object], finish_reason: str | None = None) -> dict[str, object]:
    return {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
    }


def _chunk(event: TextStreamEvent, patch: dict[str, object]) -> str:
    payload: dict[str, object] = {
        "id": event.response_id,
        "object": "chat.completion.chunk",
        "created": event.created,
        "model": event.model,
    }
    payload.update(patch)
    return "data: " + safe_json_dumps(payload) + "\n\n"


__all__ = [
    "internal_to_openai_chat_completions_response",
    "openai_chat_completions_to_internal",
    "serialize_openai_usage",
    "serialize_openai_chat_completions_stream_event",
]
