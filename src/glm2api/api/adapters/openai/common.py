"""Shared parsing helpers for the OpenAI public protocol adapters."""

from __future__ import annotations

import mimetypes

from ....core.models import ToolChoice


def file_data_to_data_url(file_data: object, filename: object = None) -> str | None:
    """Normalize OpenAI file data for the upload pipeline."""
    if not isinstance(file_data, str) or not file_data.strip():
        return None

    data_url = file_data.strip()
    if data_url.lower().startswith("data:"):
        return data_url

    mime_type = (
        mimetypes.guess_type(filename.strip())[0]
        if isinstance(filename, str) and filename.strip()
        else None
    ) or "application/octet-stream"
    return f"data:{mime_type};base64,{data_url}"


def tool_choice_from_openai(value: object) -> ToolChoice | None:
    """Parse the shared tool-choice shapes accepted by OpenAI endpoints."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"auto", "required", "none"}:
            return ToolChoice(mode=normalized)
        raise ValueError(f"OpenAI tool_choice 值不支持: {value}")
    if not isinstance(value, dict):
        raise ValueError("OpenAI tool_choice 必须是字符串或对象")

    choice_type = str(value.get("type", "")).strip().lower()
    if choice_type in {"auto", "required", "none"}:
        return ToolChoice(mode=choice_type)
    if choice_type in {"function", "tool"}:
        function = value.get("function")
        name = function.get("name") if isinstance(function, dict) else value.get("name")
        name = str(name or "").strip()
        if name:
            return ToolChoice(mode="function", name=name)
    raise ValueError("OpenAI tool_choice 对象缺少有效的 type/name")


__all__ = ["file_data_to_data_url", "tool_choice_from_openai"]
