"""Protocol-neutral request and result models used between adapters and transports.

The public APIs have different field names and content shapes.  These small
models keep that difference at the edge instead of making the ChatGLM client
depend on one public protocol's dictionary layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .usage import TokenUsage


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """A protocol-neutral text, image, file, or preserved raw content block."""

    kind: str
    text: str | None = None
    url: str | None = None
    detail: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ToolCall:
    """A normalized function call emitted by an assistant message."""

    id: str
    name: str
    arguments: str = "{}"

@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A normalized function tool definition."""

    name: str
    description: str = ""
    parameters: dict[str, object] = field(default_factory=dict)
    strict: bool | None = None

@dataclass(frozen=True, slots=True)
class ToolChoice:
    """Protocol-neutral tool selection policy."""

    mode: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "required", "none", "function"}:
            raise ValueError(f"unsupported tool choice mode: {self.mode!r}")
        if self.mode == "function" and not self.name:
            raise ValueError("function tool choice requires a name")



MessageContent = str | tuple[ContentBlock, ...] | None


@dataclass(frozen=True, slots=True)
class Message:
    """A protocol-neutral conversation message."""

    role: str
    content: MessageContent = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None



@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A partial tool call emitted while a text response is streaming."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True, slots=True)
class TextStreamEvent:
    """Protocol-neutral text-generation stream event."""

    kind: str
    response_id: str = ""
    model: str = ""
    created: int = 0
    role: str | None = None
    text: str = ""
    reasoning_content: str = ""
    tool_call: ToolCallDelta | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class TextGenerationResponse:
    """Protocol-neutral assistant result."""

    response_id: str
    model: str
    created: int
    message: Message
    finish_reason: str
    usage: TokenUsage


@dataclass(slots=True)
class TextGenerationRequest:
    """Canonical text-generation request shared by all text protocols."""

    model: str
    messages: tuple[Message, ...]
    stream: bool = False
    # Canonical output-token budget; protocol adapters map their public field
    # (max_tokens, max_completion_tokens, or max_output_tokens) here.
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | tuple[str, ...] | None = None
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice | None = None
    response_format: dict[str, object] | None = None
    reasoning_effort: str | None = None
    web_search: bool = False
    deep_research: bool = False
    extra: dict[str, object] = field(default_factory=dict)
    usage: TokenUsage | None = field(default=None, repr=False, compare=False)

__all__ = [
    "ContentBlock",
    "Message",
    "TextGenerationRequest",
    "TextGenerationResponse",
    "TextStreamEvent",
    "ToolCall",
    "ToolCallDelta",
    "ToolChoice",
    "ToolDefinition",
]
