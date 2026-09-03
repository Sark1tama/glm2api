"""OpenAI Responses API (/v1/responses) adapter.

Converts between the public Responses format and the protocol-neutral text
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
from ....core.usage import TokenUsage
from ...sse import CLIENT_DISCONNECTED, SSEWriter
from .common import file_data_to_data_url, tool_choice_from_openai


def _safe_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_responses_sse_error(
    writer: SSEWriter,
    accumulator,
    message: str,
    error_type: str,
    *,
    logger: Logger,
    path: str,
) -> None:
    """Write a Responses-specific error event using its accumulator envelope."""
    event = accumulator.sse(
        "error",
        {
            "type": "error",
            "code": error_type,
            "message": message,
            "param": None,
        },
    )
    try:
        writer.write(event)
    except CLIENT_DISCONNECTED:
        logger.warning("客户端在 Responses SSE 错误写回前断开 path=%s", path)


def serialize_openai_responses_usage(
    usage: TokenUsage,
    *,
    include_output_details: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
    if include_output_details:
        result["output_tokens_details"] = {"reasoning_tokens": 0}
    return result


_RESPONSES_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant"})


def _response_part_to_internal(part: dict[str, object], path: str = "content") -> ContentBlock:
    part_type = part.get("type")
    if part_type in {"input_text", "output_text", "text"}:
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Responses {path}.text 必须是字符串")
        return ContentBlock(kind="text", text=text)
    if part_type in {"input_image", "image_url"}:
        image_url = part.get("image_url") or part.get("url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str) or not image_url.strip():
            raise ValueError(f"Responses {path}.image_url 必须是非空 URL")
        detail = part.get("detail")
        return ContentBlock(kind="image", url=str(image_url), detail=str(detail) if detail else None)
    if part_type in {"input_file", "file"}:
        file_url = part.get("file_url")
        if isinstance(file_url, dict):
            file_url = file_url.get("url")
        if isinstance(file_url, str) and file_url.strip():
            return ContentBlock(kind="file", url=file_url.strip())

        file_data = part.get("file_data")
        data_url = file_data_to_data_url(file_data, part.get("filename"))
        if data_url:
            return ContentBlock(kind="file", url=data_url)

        file_id = part.get("file_id")
        if file_id is not None and str(file_id).strip():
            raise ValueError(
                f"Responses {path}.file_id 无法由本地代理解析，请改用 file_url 或 file_data"
            )
        raise ValueError(f"Responses {path} 必须包含 file_url 或 file_data")
    raise ValueError(f"Responses {path} 暂不支持 content 类型: {part_type}")


def _response_content_to_internal(
    content: object,
    path: str = "content",
) -> str | tuple[ContentBlock, ...] | None:
    if isinstance(content, str):
        return content
    if content is None:
        raise ValueError(f"Responses {path} 缺失")
    if not isinstance(content, list):
        raise ValueError(f"Responses {path} 必须是字符串或 block 数组")

    parts: list[ContentBlock] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise ValueError(f"Responses {path}[{index}] 必须是对象")
        converted = _response_part_to_internal(part, f"{path}[{index}]")
        parts.append(converted)
    if len(parts) == 1 and parts[0].kind == "text":
        return parts[0].text or ""
    if parts:
        return tuple(parts)
    return ""


def _response_tool_output_to_internal(output: object, path: str) -> str | tuple[ContentBlock, ...]:
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    if isinstance(output, list):
        parts: list[ContentBlock] = []
        for index, raw_part in enumerate(output):
            if not isinstance(raw_part, dict):
                raise ValueError(f"Responses {path}[{index}] 必须是对象")
            converted = _response_part_to_internal(raw_part, f"{path}[{index}]")
            if converted is None:
                raise ValueError(
                    f"Responses {path}[{index}] 暂不支持 content 类型: {raw_part.get('type')}"
                )
            parts.append(converted)
        if not parts:
            return ""
        if len(parts) == 1 and parts[0].kind == "text":
            return parts[0].text or ""
        return tuple(parts)
    try:
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Responses {path} 必须是字符串、内容数组或 JSON 值") from exc


def _append_internal_message(messages: list[Message], item: dict[str, object], path: str) -> None:
    role = item.get("role")
    if not isinstance(role, str) or role not in _RESPONSES_MESSAGE_ROLES:
        raise ValueError(f"Responses {path}.role 不受支持")
    converted_content = _response_content_to_internal(item.get("content"), path)
    messages.append(Message(role=role, content=converted_content))


# ---------------------------------------------------------------------------
# Request conversion: Responses -> internal text model
# ---------------------------------------------------------------------------


def openai_responses_to_internal(payload: dict[str, object]) -> TextGenerationRequest:
    """Convert an OpenAI Responses request body to the internal text model."""
    if not isinstance(payload, dict):
        raise ValueError("Responses 请求体顶层必须是对象")
    messages: list[Message] = []

    # --- instructions -> system ---
    instructions = payload.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValueError("Responses instructions 必须是字符串")
        if instructions:
            messages.append(Message(role="system", content=instructions))

    # --- input ---
    input_data = payload.get("input")
    if isinstance(input_data, str):
        messages.append(Message(role="user", content=input_data))
    elif isinstance(input_data, list):
        for item_index, item in enumerate(input_data):
            if not isinstance(item, dict):
                raise ValueError(f"Responses input[{item_index}] 必须是对象")
            item_type = item.get("type")

            if item_type == "message" or (item_type is None and "content" in item):
                _append_internal_message(messages, item, f"input[{item_index}].content")

            elif item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Responses input[{item_index}].call_id 不能为空")
                call_id = call_id.strip()
                tool_name = ""
                for prev_msg in reversed(messages):
                    if prev_msg.role == "assistant" and prev_msg.tool_calls:
                        for tool_call in prev_msg.tool_calls:
                            if tool_call.id == call_id:
                                tool_name = tool_call.name
                                break
                        if tool_name:
                            break
                messages.append(
                    Message(
                        role="tool",
                        content=_response_tool_output_to_internal(
                            item.get("output"),
                            f"input[{item_index}].output",
                        ),
                        tool_call_id=call_id,
                        name=tool_name or None,
                    )
                )

            elif item_type == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Responses input[{item_index}].call_id 不能为空")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Responses input[{item_index}].name 不能为空")
                raw_arguments = item.get("arguments", "{}")
                if not isinstance(raw_arguments, (str, dict)):
                    raise ValueError(f"Responses input[{item_index}].arguments 必须是 JSON 字符串或对象")
                try:
                    args = json.dumps(raw_arguments, ensure_ascii=False, separators=(",", ":")) \
                        if isinstance(raw_arguments, dict) else raw_arguments
                except (TypeError, ValueError):
                    raise ValueError(f"Responses input[{item_index}].arguments 无法序列化") from None
                tool_call = ToolCall(
                    id=call_id.strip(),
                    name=name.strip(),
                    arguments=args,
                )
                if messages and messages[-1].role == "assistant" and messages[-1].tool_calls:
                    messages[-1] = Message(
                        role=messages[-1].role,
                        content=messages[-1].content,
                        tool_calls=messages[-1].tool_calls + (tool_call,),
                        tool_call_id=messages[-1].tool_call_id,
                        name=messages[-1].name,
                        reasoning_content=messages[-1].reasoning_content,
                    )
                else:
                    messages.append(Message(role="assistant", content=None, tool_calls=(tool_call,)))

    elif input_data is not None:
        raise ValueError("Responses input 必须是字符串或数组")

    # --- tools ---
    resp_tools = payload.get("tools")
    tools: list[ToolDefinition] = []
    web_search = False
    if isinstance(resp_tools, list) and resp_tools:
        for index, tool in enumerate(resp_tools):
            if not isinstance(tool, dict):
                raise ValueError(f"Responses tools[{index}] 必须是对象")
            if tool.get("type") == "function":
                name = str(tool.get("name", "")).strip()
                if not name:
                    raise ValueError(f"Responses tools[{index}].name 不能为空")
                parameters = tool.get("parameters", {})
                if not isinstance(parameters, dict):
                    raise ValueError(f"Responses tools[{index}].parameters 必须是对象")
                tools.append(
                    ToolDefinition(
                        name=name,
                        description=str(tool.get("description", "") or ""),
                        parameters=dict(parameters),
                        strict=tool["strict"] if isinstance(tool.get("strict"), bool) else None,
                    )
                )
            elif str(tool.get("type", "")).startswith("web_search"):
                web_search = True
            else:
                raise ValueError(f"Responses tools[{index}].type 暂不支持: {tool.get('type')}")
    elif resp_tools is not None:
        raise ValueError("Responses tools 必须是数组")

    # --- reasoning ---
    reasoning = payload.get("reasoning")
    reasoning_effort: str | None = None
    if reasoning is not None and not isinstance(reasoning, dict):
        raise ValueError("Responses reasoning 必须是对象")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort is not None:
            if not isinstance(effort, str) or not effort.strip():
                raise ValueError("Responses reasoning.effort 必须是非空字符串")
            reasoning_effort = effort.strip()

    max_output_tokens = payload.get("max_output_tokens")
    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        raise ValueError("Responses stream 必须是布尔值")
    if max_output_tokens is not None and (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 0
    ):
        raise ValueError("Responses max_output_tokens 必须是非负整数")
    for field_name, value in (("temperature", temperature), ("top_p", top_p)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"Responses {field_name} 必须是数字")
    return TextGenerationRequest(
        model=str(payload.get("model") or DEFAULT_CHAT_MODEL_NAME),
        messages=tuple(messages),
        stream=bool(payload.get("stream", False)),
        max_tokens=max_output_tokens if isinstance(max_output_tokens, int) else None,
        temperature=temperature if isinstance(temperature, (int, float)) else None,
        top_p=top_p if isinstance(top_p, (int, float)) else None,
        tools=tuple(tool for tool in tools if tool.name),
        tool_choice=tool_choice_from_openai(payload.get("tool_choice")),
        reasoning_effort=reasoning_effort,
        web_search=web_search,
    )


# ---------------------------------------------------------------------------
# Non-streaming response: internal -> Responses
# ---------------------------------------------------------------------------


def internal_to_openai_responses_response(
    result: TextGenerationResponse,
    model: str,
    *,
    max_output_tokens: int | None = None,
) -> dict[str, object]:
    """Convert an internal text result to Responses format."""
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    output: list[dict[str, object]] = []
    output_text_parts: list[str] = []
    message = result.message
    finish_reason = result.finish_reason
    usage = result.usage
    status = "incomplete" if finish_reason == "length" else "completed"
    incomplete_details = {"reason": "max_output_tokens"} if status == "incomplete" else None

    # Build output message item
    msg_content: list[dict[str, object]] = []
    if isinstance(message.content, str) and message.content:
        output_text_parts.append(message.content)
        msg_content.append({
            "type": "output_text",
            "text": message.content,
            "annotations": [],
        })
    elif isinstance(message.content, tuple):
        text = "".join(block.text or "" for block in message.content if block.kind == "text")
        if text:
            output_text_parts.append(text)
            msg_content.append({
                "type": "output_text",
                "text": text,
                "annotations": [],
            })

    if msg_content:
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "status": status,
            "role": "assistant",
            "content": msg_content,
        })

    # Tool calls -> function_call items
    for tool_call in message.tool_calls:
        output.append({
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "call_id": tool_call.id or f"call_{uuid.uuid4().hex[:24]}",
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "status": "completed",
        })

    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": None,
        "incomplete_details": incomplete_details,
        "instructions": None,
        "max_output_tokens": max_output_tokens,
        "model": model,
        "output": output,
        "output_text": "".join(output_text_parts),
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "store": False,
        "usage": serialize_openai_responses_usage(usage),
    }


# ---------------------------------------------------------------------------
# Streaming: internal events -> Responses SSE
# ---------------------------------------------------------------------------


class OpenAIResponsesStreamAccumulator:
    """Serialize internal stream events into Responses SSE events."""

    def __init__(
        self,
        model: str,
        usage: TokenUsage | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.started = False
        self.output_index = 0
        self.content_index = 0
        self.current_type: str | None = None  # "text" or "function_call"
        self.usage = usage or TokenUsage()
        self.max_output_tokens = max_output_tokens
        self.finish_reason = "stop"
        self._text_buffer = ""
        self._full_text = ""  # accumulated full text for message done event
        self._current_msg_id: str | None = None
        self._current_fc_id: str | None = None
        self._pending_tool_calls: dict[int, dict[str, str]] = {}
        self._message_started = False
        self._content_part_started = False
        self._completed_output: list[dict[str, object]] = []
        self._finished = False
        self.sequence_number = 0

    def _base_response(self, status: str = "in_progress") -> dict[str, object]:
        usage: dict[str, object] | None = None
        if status in {"completed", "incomplete"}:
            usage = serialize_openai_responses_usage(self.usage, include_output_details=True)
        incomplete_details = {"reason": "max_output_tokens"} if status == "incomplete" else None
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "completed_at": int(time.time()) if status == "completed" else None,
            "error": None,
            "incomplete_details": incomplete_details,
            "instructions": None,
            "max_output_tokens": self.max_output_tokens,
            "model": self.model,
            "output": list(self._completed_output),
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "store": False,
            "temperature": 1,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1,
            "truncation": "disabled",
            "usage": usage,
            "user": None,
            "metadata": {},
        }

    def start_response(self) -> list[str]:
        self.started = True
        events: list[str] = []
        events.append(self.sse("response.created", self._base_response()))
        events.append(self.sse("response.in_progress", self._base_response()))
        return events

    def feed_event(self, event: TextStreamEvent) -> list[str]:
        """Process a protocol-neutral stream event."""
        events: list[str] = []
        if not self.started:
            events.extend(self.start_response())
        if event.usage is not None:
            self.usage = event.usage

        if event.kind == "text_delta" and event.text:
            if not self._message_started:
                events.extend(self._start_message_output())
            if not self._content_part_started:
                events.extend(self._start_content_part())
            events.append(self.sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": self._current_msg_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "delta": event.text,
            }))
            self._text_buffer += event.text
            self._full_text += event.text

        elif event.kind == "tool_call_delta" and event.tool_call is not None:
            delta = event.tool_call
            if delta.index not in self._pending_tool_calls:
                if self._content_part_started:
                    events.extend(self._end_content_part())
                if self._message_started:
                    events.extend(self._end_message_output())

                fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                call_id = delta.id or f"call_{uuid.uuid4().hex[:24]}"
                tool_name = delta.name or ""
                self._pending_tool_calls[delta.index] = {
                    "id": fc_id,
                    "call_id": call_id,
                    "name": tool_name,
                    "arguments": "",
                    "output_index": str(self.output_index),
                }
                self._current_fc_id = fc_id
                self.current_type = "function_call"
                fc_item: dict[str, object] = {
                    "type": "function_call",
                    "id": fc_id,
                    "call_id": call_id,
                    "name": tool_name,
                    "arguments": "",
                    "status": "in_progress",
                }
                events.append(self.sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": self.output_index,
                    "item": fc_item,
                }))
                self.output_index += 1

            if delta.arguments:
                tool_data = self._pending_tool_calls[delta.index]
                tool_data["arguments"] += delta.arguments
                events.append(self.sse("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": tool_data["id"],
                    "output_index": int(tool_data["output_index"]),
                    "delta": delta.arguments,
                }))

        elif event.kind == "finish":
            if event.finish_reason == "length":
                self.finish_reason = "length"
            self._complete_pending_tool_calls(events)
            events.extend(self.finish())

        elif event.kind == "done":
            events.extend(self.finish())

        return events

    def _complete_pending_tool_calls(self, events: list[str]) -> None:
        for tool_data in self._pending_tool_calls.values():
            output_index = int(tool_data["output_index"])
            events.append(self.sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": tool_data["id"],
                "output_index": output_index,
                "arguments": tool_data["arguments"],
            }))
            completed: dict[str, object] = {
                "type": "function_call",
                "id": tool_data["id"],
                "call_id": tool_data["call_id"],
                "name": tool_data["name"],
                "arguments": tool_data["arguments"],
                "status": "completed",
            }
            self._completed_output.append(completed)
            events.append(self.sse("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": completed,
            }))
        self._pending_tool_calls.clear()

    def _start_message_output(self) -> list[str]:
        self._current_msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._message_started = True
        self.current_type = "text"
        msg_item: dict[str, object] = {
            "type": "message",
            "id": self._current_msg_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        return [self.sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": self.output_index,
            "item": msg_item,
        })]

    def _start_content_part(self) -> list[str]:
        self._content_part_started = True
        part: dict[str, object] = {"type": "output_text", "text": "", "annotations": []}
        return [self.sse("response.content_part.added", {
            "type": "response.content_part.added",
            "item_id": self._current_msg_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "part": part,
        })]

    def _end_content_part(self) -> list[str]:
        events: list[str] = []
        events.append(self.sse("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": self._current_msg_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "text": self._text_buffer,
        }))
        events.append(self.sse("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": self._current_msg_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "part": {"type": "output_text", "text": self._text_buffer, "annotations": []},
        }))
        self._content_part_started = False
        self.content_index += 1
        self._text_buffer = ""
        return events

    def _end_message_output(self) -> list[str]:
        events: list[str] = []
        msg_done: dict[str, object] = {
            "type": "message",
            "id": self._current_msg_id,
            "status": "incomplete" if self.finish_reason == "length" else "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": self._full_text, "annotations": []}] if self._full_text else [],
        }
        self._completed_output.append(msg_done)
        events.append(self.sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": self.output_index,
            "item": msg_done,
        }))
        self._message_started = False
        self.output_index += 1
        self.content_index = 0
        return events

    def finish(self) -> list[str]:
        """Emit the terminal Responses events once the upstream stream ends."""
        if self._finished:
            return []
        self._finished = True
        events: list[str] = []
        if self._content_part_started:
            events.extend(self._end_content_part())
        if self._message_started:
            events.extend(self._end_message_output())

        for tc_idx, tc_data in self._pending_tool_calls.items():
            tc_out_idx = tc_data["output_index"]
            events.append(self.sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": tc_data["id"],
                "output_index": tc_out_idx,
                "arguments": tc_data["arguments"],
            }))
            fc_done: dict[str, object] = {
                "type": "function_call",
                "id": tc_data["id"],
                "call_id": tc_data["call_id"],
                "name": tc_data["name"],
                "arguments": tc_data["arguments"],
                "status": "completed",
            }
            self._completed_output.append(fc_done)
            events.append(self.sse("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": tc_out_idx,
                "item": fc_done,
            }))
        self._pending_tool_calls.clear()

        status = "incomplete" if self.finish_reason == "length" else "completed"
        terminal_event = "response.incomplete" if status == "incomplete" else "response.completed"
        events.append(self.sse(terminal_event, self._base_response(status)))
        events.append("data: [DONE]\n\n")
        return events

    def sse(self, event_type: str, data: dict[str, object]) -> str:
        if data.get("object") == "response":
            event_payload: dict[str, object] = {"type": event_type, "response": data, "response_id": self.response_id}
        else:
            event_payload = dict(data)
            event_payload["type"] = event_type
            event_payload.setdefault("response_id", self.response_id)
        event_payload["model"] = self.model
        event_payload["sequence_number"] = self.sequence_number
        self.sequence_number += 1
        return f"event: {event_type}\ndata: {_safe_json(event_payload)}\n\n"
