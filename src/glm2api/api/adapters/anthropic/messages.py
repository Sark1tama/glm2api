"""Anthropic Messages API (/v1/messages) adapter.

Converts between Anthropic Messages format and the protocol-neutral text
request model used by the shared GLM pipeline.
"""

from __future__ import annotations

import json
import time
import uuid
from logging import Logger

from ....config import DEFAULT_CHAT_MODEL_NAME
from ....core.models import (
    ContentBlock,
    Message,
    TextGenerationRequest,
    TextGenerationResponse,
    TextStreamEvent,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)
from ....core.usage import TokenUsage, estimate_conservative_prompt_tokens
from ...sse import CLIENT_DISCONNECTED, SSEWriter


_SUPPORTED_ANTHROPIC_MESSAGE_BLOCK_TYPES = frozenset(
    {"text", "image", "document", "thinking", "redacted_thinking", "tool_use", "tool_result"}
)
_SUPPORTED_ANTHROPIC_MESSAGE_ROLES = frozenset({"user", "assistant"})

_ANTHROPIC_NON_INPUT_FIELDS = frozenset(
    {
        "model",
        "stream",
        "max_tokens",
        "max_output_tokens",
        "temperature",
        "top_p",
        "stop",
        "stop_sequences",
        "metadata",
    }
)


def _safe_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def serialize_anthropic_usage(usage: TokenUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def estimate_anthropic_input_tokens(payload: object) -> int:
    """Estimate input tokens from a raw Anthropic request body.

    Count-token requests retain the complete raw block payload (including
    document metadata), so estimation deliberately happens before
    ``anthropic_messages_to_internal``.  The shared estimator is conservative and the
    request-control fields are excluded from the prompt payload.
    """
    if not isinstance(payload, dict):
        return estimate_conservative_prompt_tokens(payload)
    countable_payload = {
        key: value
        for key, value in payload.items()
        if key not in _ANTHROPIC_NON_INPUT_FIELDS
    }
    return estimate_conservative_prompt_tokens(countable_payload)


def count_anthropic_cache_control_markers(value: object) -> int:
    """Count cache-control markers without applying caching semantics."""
    if isinstance(value, dict):
        return sum(
            (1 if key == "cache_control" else 0) + count_anthropic_cache_control_markers(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(count_anthropic_cache_control_markers(item) for item in value)
    return 0


def write_anthropic_sse_error(
    writer: SSEWriter,
    message: str,
    error_type: str,
    *,
    logger: Logger,
    path: str,
) -> None:
    """Write an Anthropic-specific error event to an SSE stream."""
    event = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }
    try:
        writer.write(f"event: error\ndata: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n")
    except CLIENT_DISCONNECTED:
        logger.warning("客户端在 Anthropic SSE 错误写回前断开 path=%s", path)


def _validated_block(block: object, path: str, supported_types: frozenset[str]) -> dict[str, object]:
    if not isinstance(block, dict):
        raise ValueError(f"Anthropic {path} 必须是对象")
    block_type = block.get("type")
    if not isinstance(block_type, str) or not block_type.strip():
        raise ValueError(f"Anthropic {path}.type 缺失或无效")
    if block_type not in supported_types:
        raise ValueError(f"Anthropic {path} 暂不支持 content block 类型: {block_type}")
    return block


def _image_content_block(block: dict[str, object], path: str) -> ContentBlock:
    source = block.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"Anthropic {path}.source 必须是对象")

    source_type = source.get("type")
    if source_type == "base64":
        data = source.get("data")
        if not isinstance(data, str) or not data:
            raise ValueError(f"Anthropic {path}.source.data 不能为空")
        media_type = str(source.get("media_type") or "image/png")
        return ContentBlock(kind="image", url=f"data:{media_type};base64,{data}")
    if source_type == "url":
        url = source.get("url")
        if isinstance(url, str) and url:
            return ContentBlock(kind="image", url=url)
    raise ValueError(f"Anthropic {path}.source 类型暂不支持: {source_type}")


def _document_content_blocks(block: dict[str, object], path: str) -> tuple[ContentBlock, ...]:
    """Convert an Anthropic document source to uploadable internal blocks."""
    def with_annotations(parts: tuple[ContentBlock, ...]) -> tuple[ContentBlock, ...]:
        annotations: list[str] = []
        title = block.get("title")
        context = block.get("context")
        if isinstance(title, str) and title.strip():
            annotations.append(f"Document title: {title.strip()}")
        if isinstance(context, str) and context.strip():
            annotations.append(context.strip())
        if annotations:
            return parts + (ContentBlock(kind="text", text="\n".join(annotations)),)
        return parts

    source = block.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"Anthropic {path}.source 必须是对象")

    source_type = source.get("type")
    if source_type == "base64":
        data = source.get("data")
        if not isinstance(data, str) or not data.strip():
            raise ValueError(f"Anthropic {path}.source.data 不能为空")
        media_type = source.get("media_type")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError(f"Anthropic {path}.source.media_type 不能为空")
        encoded = data.strip()
        data_url = encoded if encoded.lower().startswith("data:") else (
            f"data:{media_type.strip()};base64,{encoded}"
        )
        return with_annotations((ContentBlock(kind="file", url=data_url),))

    if source_type == "url":
        url = source.get("url")
        if isinstance(url, str) and url.strip():
            return with_annotations((ContentBlock(kind="file", url=url.strip()),))
        raise ValueError(f"Anthropic {path}.source.url 不能为空")

    if source_type == "text":
        data = source.get("data")
        if isinstance(data, str):
            return with_annotations((ContentBlock(kind="text", text=data),))
        raise ValueError(f"Anthropic {path}.source.data 必须是字符串")

    if source_type == "content":
        content = source.get("content")
        if isinstance(content, str):
            return with_annotations((ContentBlock(kind="text", text=content),))
        if not isinstance(content, list):
            raise ValueError(f"Anthropic {path}.source.content 必须是字符串或 block 数组")

        parts: list[ContentBlock] = []
        for index, raw_part in enumerate(content):
            part_path = f"{path}.source.content[{index}]"
            part = _validated_block(raw_part, part_path, frozenset({"text", "image"}))
            if part.get("type") == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"Anthropic {part_path}.text 必须是字符串")
                parts.append(ContentBlock(kind="text", text=text))
            else:
                parts.append(_image_content_block(part, part_path))
        if parts:
            return with_annotations(tuple(parts))
        raise ValueError(f"Anthropic {path}.source.content 不能为空")

    if source_type == "file":
        file_id = source.get("file_id")
        if file_id is not None and str(file_id).strip():
            raise ValueError(
                f"Anthropic {path}.source.file_id 无法由本地代理解析，请改用 base64、url、text 或 content"
            )

    raise ValueError(f"Anthropic {path}.source 类型暂不支持: {source_type}")


def _tool_result_content(content: object, path: str) -> str | tuple[ContentBlock, ...]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        if content is None:
            return ""
        raise ValueError(f"Anthropic {path}.content 必须是字符串或文本/图片 block 数组")

    parts: list[ContentBlock] = []
    for index, raw_block in enumerate(content):
        block_path = f"{path}.content[{index}]"
        block = _validated_block(raw_block, block_path, frozenset({"text", "image"}))
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Anthropic {block_path}.text 必须是字符串")
            parts.append(ContentBlock(kind="text", text=text))
        else:
            parts.append(_image_content_block(block, block_path))
    if len(parts) == 1 and parts[0].kind == "text":
        return parts[0].text or ""
    return tuple(parts)


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> internal text model
# ---------------------------------------------------------------------------


def anthropic_messages_to_internal(payload: dict[str, object]) -> TextGenerationRequest:
    """Convert an Anthropic Messages request body to the internal text model."""
    if not isinstance(payload, dict):
        raise ValueError("Anthropic 请求体顶层必须是对象")
    messages: list[Message] = []

    # --- system ---
    system = payload.get("system")
    if system is not None:
        if isinstance(system, str):
            messages.append(Message(role="system", content=system))
        elif isinstance(system, list):
            text_parts = []
            for index, raw_block in enumerate(system):
                block = _validated_block(raw_block, f"system[{index}]", frozenset({"text"}))
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"Anthropic system[{index}].text 必须是字符串")
                text_parts.append(text)
            if text_parts:
                messages.append(Message(role="system", content="\n".join(text_parts)))
        else:
            raise ValueError("Anthropic system 必须是字符串或文本 block 数组")

    # --- messages ---
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("Anthropic messages 必须是数组")
    for message_index, msg in enumerate(raw_messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Anthropic messages[{message_index}] 必须是对象")
        role = msg.get("role")
        if not isinstance(role, str) or role not in _SUPPORTED_ANTHROPIC_MESSAGE_ROLES:
            raise ValueError(f"Anthropic messages[{message_index}].role 必须是 user 或 assistant")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append(Message(role=role, content=content))
            continue

        if content is None:
            messages.append(Message(role=role, content=""))
            continue
        if not isinstance(content, list):
            raise ValueError(f"Anthropic messages[{message_index}].content 必须是字符串或 block 数组")

        # Process content blocks in their original order.  The internal model
        # represents tool calls/results as messages, so mixed Anthropic blocks
        # are split into ordered messages instead of being flattened.
        content_parts: list[ContentBlock] = []
        pending_tool_calls: list[ToolCall] = []
        emitted_message = False

        def flush_tool_calls() -> None:
            nonlocal emitted_message
            if not pending_tool_calls:
                return
            messages.append(
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=tuple(pending_tool_calls),
                )
            )
            pending_tool_calls.clear()
            emitted_message = True

        def flush_content() -> None:
            nonlocal emitted_message
            if not content_parts:
                return
            if len(content_parts) == 1 and content_parts[0].kind == "text":
                messages.append(Message(role=role, content=content_parts[0].text or ""))
            else:
                messages.append(Message(role=role, content=tuple(content_parts)))
            content_parts.clear()
            emitted_message = True

        for block_index, raw_block in enumerate(content):
            block = _validated_block(
                raw_block,
                f"messages[{message_index}].content[{block_index}]",
                _SUPPORTED_ANTHROPIC_MESSAGE_BLOCK_TYPES,
            )
            block_type = block.get("type")

            if block_type == "text":
                flush_tool_calls()
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"Anthropic messages[{message_index}].content[{block_index}].text 必须是字符串"
                    )
                content_parts.append(ContentBlock(kind="text", text=text))

            elif block_type == "thinking":
                flush_tool_calls()
                thinking_text = block.get("thinking")
                if not isinstance(thinking_text, str):
                    raise ValueError(
                        f"Anthropic messages[{message_index}].content[{block_index}].thinking 必须是字符串"
                    )
                metadata = dict(block)
                metadata.pop("type", None)
                metadata.pop("thinking", None)
                content_parts.append(
                    ContentBlock(
                        kind="thinking",
                        text=thinking_text,
                        metadata=metadata,
                    )
                )

            elif block_type == "redacted_thinking":
                flush_tool_calls()
                data = block.get("data")
                if not isinstance(data, str) or not data:
                    raise ValueError(
                        f"Anthropic messages[{message_index}].content[{block_index}].data 不能为空"
                    )
                metadata = dict(block)
                metadata.pop("type", None)
                content_parts.append(ContentBlock(kind="redacted_thinking", metadata=metadata))

            elif block_type == "image":
                flush_tool_calls()
                content_parts.append(
                    _image_content_block(
                        block,
                        f"messages[{message_index}].content[{block_index}]",
                    )
                )

            elif block_type == "document":
                flush_tool_calls()
                content_parts.extend(
                    _document_content_blocks(
                        block,
                        f"messages[{message_index}].content[{block_index}]",
                    )
                )

            elif block_type == "tool_use":
                flush_content()
                tool_id = block.get("id")
                tool_name = block.get("name")
                tool_input = block.get("input")
                if not isinstance(tool_id, str) or not tool_id.strip():
                    raise ValueError(
                        f"Anthropic messages[{message_index}].content[{block_index}].id 不能为空"
                    )
                if not isinstance(tool_name, str) or not tool_name.strip():
                    raise ValueError(
                        f"Anthropic messages[{message_index}].content[{block_index}].name 不能为空"
                    )
                if not isinstance(tool_input, dict):
                    raise ValueError(
                        f"Anthropic messages[{message_index}].content[{block_index}].input 必须是对象"
                    )
                pending_tool_calls.append(
                    ToolCall(
                        id=tool_id.strip(),
                        name=tool_name.strip(),
                        arguments=json.dumps(tool_input, ensure_ascii=False, separators=(",", ":")),
                    )
                )

            elif block_type == "tool_result":
                flush_content()
                flush_tool_calls()
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str) or not tool_use_id.strip():
                    raise ValueError(
                        f"Anthropic messages[{message_index}].content[{block_index}].tool_use_id 不能为空"
                    )
                result_content = block.get("content")
                result_content = _tool_result_content(
                    result_content,
                    f"messages[{message_index}].content[{block_index}]",
                )
                messages.append(
                    Message(
                        role="tool",
                        content=result_content,
                        tool_call_id=tool_use_id.strip(),
                    )
                )
                emitted_message = True

        flush_content()
        flush_tool_calls()
        if not emitted_message:
            messages.append(Message(role=role, content=""))

    tools: list[ToolDefinition] = []
    web_search = False

    # --- tools ---
    anthropic_tools = payload.get("tools")
    if isinstance(anthropic_tools, list) and anthropic_tools:
        for index, tool in enumerate(anthropic_tools):
            if not isinstance(tool, dict):
                raise ValueError(f"Anthropic tools[{index}] 必须是对象")
            tool_type = str(tool.get("type", "")).strip()
            # Anthropic server tools: web_search_20250305, web_search_20260209, web_search_20260318
            if tool_type.startswith("web_search_"):
                web_search = True
                continue
            if tool_type:
                raise ValueError(f"Anthropic tools[{index}] 暂不支持服务端工具类型: {tool_type}")
            name = str(tool.get("name", "")).strip()
            if not name:
                raise ValueError(f"Anthropic tools[{index}].name 不能为空")
            parameters = tool.get("input_schema")
            if not isinstance(parameters, dict):
                raise ValueError(f"Anthropic tools[{index}].input_schema 必须是对象")
            tools.append(
                ToolDefinition(
                    name=name,
                    description=str(tool.get("description", "") or ""),
                    parameters=dict(parameters),
                )
            )
    elif anthropic_tools is not None:
        raise ValueError("Anthropic tools 必须是数组")

    tool_choice = payload.get("tool_choice")
    normalized_tool_choice: ToolChoice | None = None
    if tool_choice is not None and not isinstance(tool_choice, dict):
        raise ValueError("Anthropic tool_choice 必须是对象")
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type", "")).strip().lower()
        if choice_type == "auto":
            normalized_tool_choice = ToolChoice(mode="auto")
        elif choice_type == "any":
            normalized_tool_choice = ToolChoice(mode="required")
        elif choice_type == "none":
            normalized_tool_choice = ToolChoice(mode="none")
        elif choice_type == "tool":
            name = str(tool_choice.get("name", "")).strip()
            if not name:
                raise ValueError("Anthropic tool_choice.type=tool 必须包含 name")
            normalized_tool_choice = ToolChoice(mode="function", name=name)
        else:
            raise ValueError(f"Anthropic tool_choice.type 暂不支持: {choice_type or '<missing>'}")

    # --- thinking ---
    thinking = payload.get("thinking")
    reasoning_effort: str | None = None
    if thinking is not None:
        if not isinstance(thinking, dict):
            raise ValueError("Anthropic thinking 必须是对象")
        thinking_type = thinking.get("type")
        if thinking_type == "enabled":
            budget_tokens = thinking.get("budget_tokens", "medium")
            if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, (int, str)):
                raise ValueError("Anthropic thinking.budget_tokens 类型无效")
            reasoning_effort = str(budget_tokens)
        elif thinking_type not in {None, "disabled"}:
            raise ValueError(f"Anthropic thinking.type 暂不支持: {thinking_type}")

    stop_sequences = payload.get("stop_sequences")
    if stop_sequences is not None and not isinstance(stop_sequences, list):
        raise ValueError("Anthropic stop_sequences 必须是数组")
    if isinstance(stop_sequences, list) and any(not isinstance(item, str) for item in stop_sequences):
        raise ValueError("Anthropic stop_sequences 的元素必须是字符串")
    stop = tuple(stop_sequences) if isinstance(stop_sequences, list) and stop_sequences else None
    max_tokens = payload.get("max_tokens")
    max_output_tokens = payload.get("max_output_tokens")
    if max_tokens is not None and max_output_tokens is not None:
        raise ValueError("Anthropic max_tokens 与 max_output_tokens 不能同时提供")
    if max_tokens is None:
        max_tokens = max_output_tokens
    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        raise ValueError("Anthropic stream 必须是布尔值")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0
    ):
        raise ValueError("Anthropic max_tokens 必须是非负整数")
    for field_name, value in (("temperature", temperature), ("top_p", top_p)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"Anthropic {field_name} 必须是数字")
    return TextGenerationRequest(
        model=str(payload.get("model") or DEFAULT_CHAT_MODEL_NAME),
        messages=tuple(messages),
        stream=bool(payload.get("stream", False)),
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        temperature=temperature if isinstance(temperature, (int, float)) else None,
        top_p=top_p if isinstance(top_p, (int, float)) else None,
        stop=stop,
        tools=tuple(tools),
        tool_choice=normalized_tool_choice,
        reasoning_effort=reasoning_effort,
        web_search=web_search,
    )


# ---------------------------------------------------------------------------
# Non-streaming response conversion: internal -> Anthropic
# ---------------------------------------------------------------------------


def internal_to_anthropic_messages_response(
    result: TextGenerationResponse,
    model: str,
) -> dict[str, object]:
    """Convert an internal text result to Anthropic Messages format."""
    content: list[dict[str, object]] = []
    stop_reason = "end_turn"
    message = result.message
    finish_reason = result.finish_reason
    usage = result.usage

    # text content
    if isinstance(message.content, str) and message.content:
        content.append({"type": "text", "text": message.content})
    elif isinstance(message.content, tuple):
        for block in message.content:
            if block.kind == "text" and block.text:
                content.append({"type": "text", "text": block.text})
            elif block.kind == "thinking":
                signature = block.metadata.get("signature")
                if isinstance(signature, str) and signature:
                    thinking_block = dict(block.metadata)
                    thinking_block["type"] = "thinking"
                    thinking_block["thinking"] = block.text or ""
                    content.append(thinking_block)
            elif block.kind == "redacted_thinking":
                data = block.metadata.get("data")
                if isinstance(data, str) and data:
                    redacted_block = dict(block.metadata)
                    redacted_block["type"] = "redacted_thinking"
                    content.append(redacted_block)

    # tool_calls
    if message.tool_calls:
        stop_reason = "tool_use"
        for tool_call in message.tool_calls:
            try:
                input_data = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError):
                input_data = {}
            content.append({
                "type": "tool_use",
                "id": tool_call.id or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": tool_call.name,
                "input": input_data,
            })

    if finish_reason == "length":
        stop_reason = "max_tokens"

    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": serialize_anthropic_usage(usage),
    }


# ---------------------------------------------------------------------------
# Streaming: internal events -> Anthropic SSE
# ---------------------------------------------------------------------------


class AnthropicMessagesStreamAccumulator:
    """Serialize internal stream events into Anthropic SSE events."""

    def __init__(
        self,
        model: str,
        usage: TokenUsage | None = None,
    ) -> None:
        self.model = model
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.started = False
        self.content_index = 0
        self.current_block_type: str | None = None
        self.usage = usage or TokenUsage()
        self.stop_reason = "end_turn"
        self._pending_tool_calls: dict[int, dict[str, object]] = {}
        self._block_open = False
        self._finished = False

    def start_message(self) -> str:
        """Emit message_start event."""
        self.started = True
        msg = {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": self.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": serialize_anthropic_usage(self.usage),
        }
        return self._sse("message_start", {"type": "message_start", "message": msg})

    def feed_event(self, event: TextStreamEvent) -> list[str]:
        """Process a protocol-neutral stream event."""
        events: list[str] = []
        if not self.started:
            events.append(self.start_message())
        if event.usage is not None:
            self.usage = event.usage

        if event.kind == "reasoning_delta" and event.reasoning_content:
            # GLM reasoning is not accompanied by an Anthropic-verifiable
            # signature.  Do not emit an invalid thinking block; retain the
            # reasoning only inside the GLM pipeline for tool-call parsing.
            pass

        elif event.kind == "text_delta" and event.text:
            if self.current_block_type != "text":
                if self._block_open:
                    events.append(self._content_block_stop())
                events.append(self._content_block_start("text", {"text": ""}))
                self.current_block_type = "text"
            events.append(self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self.content_index,
                "delta": {"type": "text_delta", "text": event.text},
            }))

        elif event.kind == "tool_call_delta" and event.tool_call is not None:
            delta = event.tool_call
            if delta.index not in self._pending_tool_calls:
                if self._block_open:
                    events.append(self._content_block_stop())
                tool_id = delta.id or f"toolu_{uuid.uuid4().hex[:24]}"
                tool_name = delta.name or ""
                self._pending_tool_calls[delta.index] = {
                    "id": tool_id,
                    "name": tool_name,
                    "arguments": "",
                }
                events.append(self._content_block_start("tool_use", {
                    "id": tool_id,
                    "name": tool_name,
                    "input": {},
                }))
                self.current_block_type = "tool_use"
                self.stop_reason = "tool_use"

            if delta.arguments:
                pending = self._pending_tool_calls[delta.index]
                pending["arguments"] = str(pending["arguments"]) + delta.arguments
                events.append(self._sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self.content_index,
                    "delta": {"type": "input_json_delta", "partial_json": delta.arguments},
                }))

        elif event.kind == "finish" and event.finish_reason:
            if event.finish_reason == "length":
                self.stop_reason = "max_tokens"
            elif event.finish_reason == "tool_calls":
                self.stop_reason = "tool_use"

        elif event.kind == "done":
            events.extend(self.finish())

        return events

    def finish(self) -> list[str]:
        """Emit the terminal Anthropic events once the upstream stream ends."""
        if self._finished:
            return []
        self._finished = True
        events: list[str] = []
        if self._block_open:
            events.append(self._content_block_stop())
        events.append(self._sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self.usage.output_tokens},
        }))
        events.append(self._sse("message_stop", {"type": "message_stop"}))
        return events

    def _content_block_start(self, block_type: str, initial: dict[str, object]) -> str:
        block: dict[str, object] = {"type": block_type}
        block.update(initial)
        self._block_open = True
        event = self._sse("content_block_start", {
            "type": "content_block_start",
            "index": self.content_index,
            "content_block": block,
        })
        return event

    def _content_block_stop(self) -> str:
        event = self._sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self.content_index,
        })
        self.content_index += 1
        self._block_open = False
        return event

    def _sse(self, event_type: str, data: dict[str, object]) -> str:
        return f"event: {event_type}\ndata: {_safe_json(data)}\n\n"
