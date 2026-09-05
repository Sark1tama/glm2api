"""Chat-specific model and mode resolution for ChatGLM."""

from __future__ import annotations

from ..config import BUILTIN_MODEL_CATALOG, BUILTIN_TEXT_MODELS


def get_model_multimodal_capability(model: str) -> bool | None:
    """Return whether a known model accepts non-text content."""
    spec = BUILTIN_MODEL_CATALOG.get((model or "").strip().lower())
    if spec is None or spec.kind != "text":
        return None
    return spec.multimodal


REASONING_EFFORT_CHAT_MODES = {
    "": "",
    "none": "",
    "off": "",
    "false": "",
    "0": "",
    "low": "",
    "minimal": "",
    "fast": "",
    "quick": "",
    "medium": "thinking",
    "balanced": "thinking",
    "think": "thinking",
    "thinking": "thinking",
    "high": "deep_thinking",
    "max": "deep_thinking",
    "maximum": "deep_thinking",
    "xhigh": "deep_thinking",
    "extreme": "deep_thinking",
    "deep": "deep_thinking",
    "deep_thinking": "deep_thinking",
}


def validate_text_model(model: str) -> str:
    """Require a public text model name before opening an upstream request."""
    requested_model = (model or "").strip()
    if requested_model not in BUILTIN_TEXT_MODELS:
        supported = ", ".join(BUILTIN_TEXT_MODELS)
        raise ValueError(
            f"当前文本接口不支持模型 {requested_model or '<empty>'}；支持的模型：{supported}。"
        )
    return requested_model


def resolve_chat_mode(reasoning_effort: object) -> str:
    if reasoning_effort is not None:
        normalized_effort = str(reasoning_effort).strip().lower().replace("-", "_")
        if normalized_effort in REASONING_EFFORT_CHAT_MODES:
            return REASONING_EFFORT_CHAT_MODES[normalized_effort]
        if normalized_effort:
            return "thinking"
        return ""
    return ""


def resolve_networking(web_search: object) -> bool:
    return bool(web_search)

__all__ = [
    "REASONING_EFFORT_CHAT_MODES",
    "get_model_multimodal_capability",
    "resolve_chat_mode",
    "resolve_networking",
    "validate_text_model",
]
