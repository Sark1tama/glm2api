"""Chat-specific model and mode resolution for ChatGLM."""

from __future__ import annotations

import re

from ..config import AppConfig


MODEL_FEATURE_SUFFIXES = ("think", "search")
MODEL_VARIANT_SUFFIXES = (
    ("think",),
    ("search",),
    ("think", "search"),
)
MODEL_MULTIMODAL_CAPABILITIES = {
    "glm-5.3": False,
    "glm-5.3-flash": True,
}


def split_model_features(model: str) -> tuple[str, set[str]]:
    parts = (model or "").strip().split("-")
    features: set[str] = set()

    while parts and parts[-1].lower() in MODEL_FEATURE_SUFFIXES:
        features.add(parts.pop().lower())

    if not features:
        return (model or "").strip(), features
    return "-".join(parts), features


def get_model_multimodal_capability(model: str) -> bool | None:
    """Return whether a known model accepts non-text content."""
    base_model, _ = split_model_features(model)
    return MODEL_MULTIMODAL_CAPABILITIES.get(base_model.lower())


def expand_model_variants(models: list[str] | tuple[str, ...], excluded_models: set[str] | None = None) -> list[str]:
    excluded = {model.lower() for model in (excluded_models or set())}
    expanded: list[str] = []

    for model in models:
        base_model, features = split_model_features(model)
        expanded.append(model)
        if features or model.lower() in excluded:
            continue
        for suffix in MODEL_VARIANT_SUFFIXES:
            expanded.append(f"{base_model}-{'-'.join(suffix)}")

    return expanded


def model_requests_thinking(model: str) -> bool:
    _, features = split_model_features(model)
    return "think" in features


def model_requests_search(model: str) -> bool:
    _, features = split_model_features(model)
    return "search" in features


ASSISTANT_ID_PATTERN = re.compile(r"^[a-z0-9]{24,}$")
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


def resolve_upstream_model(requested_model: str, config: AppConfig) -> tuple[str, str]:
    base_model, _ = split_model_features(requested_model)
    upstream_model = config.model_aliases.get(base_model, base_model)
    assistant_id = upstream_model if ASSISTANT_ID_PATTERN.fullmatch(upstream_model) else config.glm_assistant_id
    return upstream_model, assistant_id


def resolve_chat_mode(model: str, reasoning_effort: object, deep_research: object) -> str:
    lower_model = (model or "").lower()
    if deep_research or "deepresearch" in lower_model or "deep-research" in lower_model:
        return "deep_thinking"
    if reasoning_effort is not None:
        normalized_effort = str(reasoning_effort).strip().lower().replace("-", "_")
        if normalized_effort in REASONING_EFFORT_CHAT_MODES:
            return REASONING_EFFORT_CHAT_MODES[normalized_effort]
        if normalized_effort:
            return "thinking"
        return ""
    if model_requests_thinking(model) or "think" in lower_model or "zero" in lower_model:
        return "thinking"
    return ""


def resolve_networking(model: str, web_search: object) -> bool:
    return bool(web_search) or model_requests_search(model)

__all__ = [
    "ASSISTANT_ID_PATTERN",
    "MODEL_FEATURE_SUFFIXES",
    "MODEL_MULTIMODAL_CAPABILITIES",
    "MODEL_VARIANT_SUFFIXES",
    "REASONING_EFFORT_CHAT_MODES",
    "expand_model_variants",
    "get_model_multimodal_capability",
    "model_requests_search",
    "model_requests_thinking",
    "resolve_chat_mode",
    "resolve_networking",
    "resolve_upstream_model",
    "split_model_features",
]
