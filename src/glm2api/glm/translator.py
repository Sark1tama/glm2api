"""Prompt and tool translation for the ChatGLM web protocol."""

from __future__ import annotations

import json
import re
from typing import Sequence

from ..core.models import (
    ContentBlock,
    Message,
    TextGenerationRequest,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)
from .tools.dsml import (
    BLOCKED_NATIVE_TOOL_NAMES,
    SERVER_SIDE_TOOL_NAMES,
    filter_tools,
    serialize_tool_call_block,
    serialize_tool_result_block,
    tools_to_prompt,
)
from ..utils.json import safe_json_dumps

URL_PATTERN = re.compile(r"https?://[^\s<>()\"']+")
POWERSHELL_CMDLET_PATTERN = re.compile(r"^[A-Z][A-Za-z]+-[A-Z][A-Za-z]+$")
POWERSHELL_ALIASES = {"cat", "cd", "copy", "del", "dir", "echo", "erase", "ls", "md", "move", "pwd", "rd", "ren", "rm", "sc", "type"}


def content_block_to_glm_payload(block: ContentBlock) -> dict[str, object]:
    """Serialize one internal content block for ChatGLM's chat wire format."""
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


def tool_call_to_glm_payload(tool_call: ToolCall) -> dict[str, object]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
        },
    }


def tool_definition_to_glm_payload(tool: ToolDefinition) -> dict[str, object]:
    function: dict[str, object] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.parameters),
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def tool_choice_to_glm_value(tool_choice: ToolChoice) -> str | dict[str, object]:
    if tool_choice.mode != "function":
        return tool_choice.mode
    return {"type": "function", "function": {"name": tool_choice.name or ""}}


def message_to_glm_payload(message: Message) -> dict[str, object]:
    if isinstance(message.content, tuple):
        if len(message.content) == 1 and message.content[0].kind == "text":
            content: object = message.content[0].text or ""
        else:
            content = [content_block_to_glm_payload(block) for block in message.content]
    else:
        content = message.content

    result: dict[str, object] = {"role": message.role, "content": content}
    if message.name is not None:
        result["name"] = message.name
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        result["tool_calls"] = [tool_call_to_glm_payload(tool_call) for tool_call in message.tool_calls]
    if message.reasoning_content is not None:
        result["reasoning_content"] = message.reasoning_content
    return result


def messages_to_glm_payload(messages: Sequence[Message]) -> list[dict[str, object]]:
    return [message_to_glm_payload(message) for message in messages]


def request_to_glm_payload(request: TextGenerationRequest) -> dict[str, object]:
    """Serialize the canonical request for provider-side diagnostics/tests."""
    result = dict(request.extra)
    result.update(
        {
            "model": request.model,
            "messages": messages_to_glm_payload(request.messages),
            "stream": request.stream,
        }
    )
    if request.max_tokens is not None:
        result["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        result["temperature"] = request.temperature
    if request.top_p is not None:
        result["top_p"] = request.top_p
    if request.stop is not None:
        result["stop"] = list(request.stop) if isinstance(request.stop, tuple) else request.stop
    if request.tools:
        result["tools"] = [tool_definition_to_glm_payload(tool) for tool in request.tools]
    if request.tool_choice is not None:
        result["tool_choice"] = tool_choice_to_glm_value(request.tool_choice)
    if request.response_format is not None:
        result["response_format"] = dict(request.response_format)
    if request.reasoning_effort is not None:
        result["reasoning_effort"] = request.reasoning_effort
    if request.web_search:
        result["web_search"] = True
    if request.deep_research:
        result["deep_research"] = True
    return result


def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text_parts.append(str(item.get("text", "")))
        elif item_type == "thinking":
            text_parts.append(str(item.get("thinking") or ""))
        elif item_type == "image_url":
            url = item.get("image_url", {}).get("url", "")
            text_parts.append(_content_reference_text("image", url))
        elif item_type == "file":
            url = item.get("file_url", {}).get("url", "")
            text_parts.append(_content_reference_text("file", url))
    return "\n".join(part for part in text_parts if part)


def _content_reference_text(kind: str, url: object) -> str:
    """Describe an attachment without copying an inline payload into text."""
    reference = str(url or "")
    if reference.lower().startswith("data:"):
        reference = reference.split(",", 1)[0]
    return f"[{kind}:{reference}]"


def extract_first_url(text: str) -> str | None:
    match = URL_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,;:!?)}+")


def extract_recent_user_url(messages: list[dict[str, object]]) -> str | None:
    for message in reversed(messages):
        if str(message.get("role", "")).strip() != "user":
            continue
        text = extract_text_content(message.get("content"))
        url = extract_first_url(text)
        if url:
            return url
    return None


def sanitize_tool_call_payload(
    tool_name: str,
    arguments: object,
    fallback_url: str | None = None,
) -> dict[str, object] | None:
    parsed_arguments = arguments
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None

    if parsed_arguments is None:
        parsed_arguments = {}
    if not isinstance(parsed_arguments, dict):
        return None

    cleaned = {str(key): value for key, value in parsed_arguments.items()}
    if cleaned == {"param_name": "url"} and fallback_url:
        cleaned = {"url": fallback_url}
    elif cleaned == {"param_name": "url"}:
        cleaned = {}
    if "param_name" in cleaned and "param_value" not in cleaned and len(cleaned) == 1:
        cleaned = {}

    if tool_name == "shell":
        command = cleaned.get("command")
        if isinstance(command, str):
            stripped_command = command.strip()
            if stripped_command.startswith("["):
                try:
                    parsed_command = json.loads(stripped_command)
                except json.JSONDecodeError:
                    parsed_command = None
                if isinstance(parsed_command, list):
                    cleaned["command"] = [str(part) for part in parsed_command]
            elif stripped_command.startswith('"'):
                try:
                    parsed_command = json.loads(f"[{stripped_command}]")
                except json.JSONDecodeError:
                    parsed_command = None
                if isinstance(parsed_command, list):
                    cleaned["command"] = [str(part) for part in parsed_command]
            else:
                cleaned["command"] = ["powershell.exe", "-Command", stripped_command]
        elif isinstance(command, list) and command:
            command_parts = [str(part) for part in command]
            command_name = command_parts[0].strip()
            lower_name = command_name.lower()
            is_shell_host = lower_name in {"powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe"}
            is_powershell_command = bool(POWERSHELL_CMDLET_PATTERN.fullmatch(command_name)) or lower_name in POWERSHELL_ALIASES
            if is_powershell_command and not is_shell_host:
                cleaned["command"] = ["powershell.exe", "-Command", " ".join(command_parts)]

    return cleaned


def sanitize_tool_calls(
    tool_calls: list[dict[str, object]],
    fallback_url: str | None = None,
) -> list[dict[str, object]]:
    sanitized: list[dict[str, object]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            continue
        tool_name = str(function.get("name", "")).strip()
        if not tool_name:
            continue
        original_arguments = function.get("arguments", "{}")
        original_value: object = original_arguments
        if isinstance(original_arguments, str):
            try:
                original_value = json.loads(original_arguments)
            except json.JSONDecodeError:
                original_value = original_arguments
        cleaned_arguments = sanitize_tool_call_payload(
            tool_name=tool_name,
            arguments=original_arguments,
            fallback_url=fallback_url,
        )
        if cleaned_arguments is None:
            continue
        repaired = not isinstance(original_value, dict) or safe_json_dumps(cleaned_arguments) != safe_json_dumps(original_value)
        sanitized.append(
            {
                "id": str(tool_call.get("id", "")) or f"call_repaired_{index}",
                "type": "function",
                "index": index,
                "_repaired": repaired,
                "function": {
                    "name": tool_name,
                    "arguments": safe_json_dumps(cleaned_arguments),
                },
            }
        )
    return sanitized


def parse_tool_choice_policy(
    tool_choice: ToolChoice | None,
    available_tool_names: set[str] | None = None,
) -> dict[str, object]:
    """Convert the canonical tool choice into the prompt policy shape."""
    if tool_choice is None:
        return {"mode": "auto", "tool_name": None}
    if tool_choice.mode == "function":
        if tool_choice.name and tool_choice.name in (available_tool_names or set()):
            return {"mode": "specific", "tool_name": tool_choice.name}
        raise ValueError(f"tool_choice 指定的工具不可用: {tool_choice.name}")
    return {"mode": tool_choice.mode, "tool_name": None}


def convert_messages_to_glm_prompt(
    messages: Sequence[Message],
    tools: Sequence[ToolDefinition] | None,
    blocked_tool_names: set[str] | None = None,
    tool_choice: ToolChoice | None = None,
    server_side_tool_names: set[str] | None = None,
) -> list[dict[str, object]]:
    chat_messages = messages_to_glm_payload(messages)
    chat_tools = [tool_definition_to_glm_payload(tool) for tool in (tools or [])]
    filtered_tools = filter_tools(chat_tools, blocked_tool_names or set())
    available_tool_names = {
        str(tool.get("function", {}).get("name", "")).strip()
        for tool in (filtered_tools or [])
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    available_tool_names.discard("")
    server_side_tool_names = server_side_tool_names or SERVER_SIDE_TOOL_NAMES
    tool_choice_policy = parse_tool_choice_policy(tool_choice, available_tool_names)
    processed: list[dict[str, str]] = []
    latest_user_url: str | None = extract_recent_user_url(chat_messages)
    valid_tool_call_ids: set[str] = set()
    repaired_tool_call_ids: set[str] = set()
    consumed_tool_result_ids: set[str] = set()
    for message in chat_messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if role == "user":
            current_text = extract_text_content(content)
            current_url = extract_first_url(current_text)
            if current_url:
                latest_user_url = current_url
        if role == "assistant" and message.get("tool_calls"):
            tool_blocks: list[str] = []
            raw_tool_calls = message.get("tool_calls", []) # pyright: ignore[reportGeneralTypeIssues]
            sanitized_tool_calls = sanitize_tool_calls(
                raw_tool_calls if isinstance(raw_tool_calls, list) else [],
                fallback_url=latest_user_url,
            )
            for tool_call in sanitized_tool_calls:
                function = tool_call.get("function", {})
                tool_name = str(function.get("name", "unknown"))
                if available_tool_names and tool_name not in available_tool_names:
                    continue
                tool_blocks.append(
                    serialize_tool_call_block(
                        name=tool_name,
                        arguments=function.get("arguments", "{}"),
                    )
                )
                tool_call_id = str(tool_call.get("id", "")).strip()
                if tool_call_id and not tool_call_id.startswith("call_repaired_"):
                    if tool_call_id in valid_tool_call_ids:
                        raise ValueError(f"assistant tool_call 的 ID 重复: {tool_call_id}")
                    valid_tool_call_ids.add(tool_call_id)
                    if bool(tool_call.get("_repaired")):
                        repaired_tool_call_ids.add(tool_call_id)
            assistant_text = extract_text_content(content).strip() if content else ""
            block = "\n".join(tool_blocks)
            if not assistant_text and not block:
                continue
            content = f"{assistant_text}\n{block}".strip() if assistant_text and block else (assistant_text or block)
        elif role == "tool":
            tool_call_id = str(message.get("tool_call_id", "")).strip()
            if not tool_call_id:
                raise ValueError("tool_result 缺少 tool_call_id")
            if tool_call_id not in valid_tool_call_ids:
                raise ValueError(f"tool_result 找不到对应的 tool_call: {tool_call_id}")
            if tool_call_id and tool_call_id in repaired_tool_call_ids:
                # The preceding assistant call was repaired and must not be replayed.
                continue
            if tool_call_id in consumed_tool_result_ids:
                raise ValueError(f"tool_result 的 tool_call_id 重复: {tool_call_id}")
            consumed_tool_result_ids.add(tool_call_id)
            role = "user"
            tool_name = str(message.get("name", "")).strip() or "unknown_tool"
            tool_result_text = extract_text_content(content)
            content = serialize_tool_result_block(
                tool_call_id=tool_call_id or message.get("tool_call_id", "unknown"),
                tool_name=tool_name,
                content=tool_result_text,
            )
        elif role == "assistant" and not content:
            continue

        text = extract_text_content(content) if content else ""
        if text:
            processed.append({"role": role, "content": text})

    transcript_parts: list[str] = []

    if filtered_tools and tool_choice_policy.get("mode") != "none":
        transcript_parts.append(
            tools_to_prompt(
                filtered_tools,
                blocked_tool_names=blocked_tool_names,
                tool_choice_policy=tool_choice_policy,
                server_side_tool_names=server_side_tool_names,
            )
        )
        transcript_parts.append("# CONVERSATION")

    for item in processed:
        title = (
            item["role"]
            .replace("system", "System")
            .replace("assistant", "Assistant")
            .replace("user", "User")
            .replace("developer", "Developer")
        )
        transcript_parts.append(f"{title}: {item['content']}".strip())

    prompt = "\n\n".join(part for part in transcript_parts if part).strip()
    return [{"role": "user", "content": [{"type": "text", "text": prompt + "\n\nAssistant: "}]}]

__all__ = [
    "BLOCKED_NATIVE_TOOL_NAMES",
    "SERVER_SIDE_TOOL_NAMES",
    "convert_messages_to_glm_prompt",
    "content_block_to_glm_payload",
    "extract_first_url",
    "extract_recent_user_url",
    "extract_text_content",
    "message_to_glm_payload",
    "messages_to_glm_payload",
    "parse_tool_choice_policy",
    "request_to_glm_payload",
    "sanitize_tool_call_payload",
    "sanitize_tool_calls",
    "tool_call_to_glm_payload",
    "tool_choice_to_glm_value",
    "tool_definition_to_glm_payload",
]
