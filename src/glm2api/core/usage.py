"""Conservative token usage estimation for the GLM web adapter.

The ChatGLM web SSE response currently does not expose token usage. These
helpers intentionally overestimate usage and must not be treated as billing
precision.
"""

from __future__ import annotations

import json
import math
import string
from dataclasses import dataclass, field
from typing import Literal


UsageSource = Literal["upstream", "estimated", "mixed", "unavailable"]
_FieldSource = Literal["upstream", "estimated", "unavailable"]


# A smaller byte budget than common BPE ratios deliberately biases estimates
# upward, especially for JSON, tool schemas, and mixed Chinese/ASCII prompts.
CONSERVATIVE_BYTES_PER_TOKEN = 1.5
PROMPT_PROTOCOL_OVERHEAD_TOKENS = 32


def _non_negative_int(value: object, default: int = 0) -> int:
    """Normalize a token count received from an untyped JSON payload."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    return default


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Protocol-neutral token usage with field-level provenance.

    The public ``source`` value is derived from the two fields.  Keeping the
    provenance internally lets an upstream value replace an estimate without
    a later estimate accidentally overwriting it.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    source: UsageSource = "unavailable"
    _input_source: _FieldSource | None = field(default=None, repr=False, compare=False)
    _output_source: _FieldSource | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_tokens", _non_negative_int(self.input_tokens))
        object.__setattr__(self, "output_tokens", _non_negative_int(self.output_tokens))
        if self.source not in {"upstream", "estimated", "mixed", "unavailable"}:
            raise ValueError(f"unsupported token usage source: {self.source!r}")

        default_field_source: _FieldSource = {
            "upstream": "upstream",
            "estimated": "estimated",
            "mixed": "estimated",
            "unavailable": "unavailable",
        }[self.source]
        input_source = self._input_source or default_field_source
        output_source = self._output_source or default_field_source
        if self.source == "mixed" and self._input_source is None and self._output_source is None:
            input_source, output_source = "upstream", "estimated"
        object.__setattr__(self, "_input_source", input_source)
        object.__setattr__(self, "_output_source", output_source)
        object.__setattr__(self, "source", self._combined_source(input_source, output_source))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def estimated(cls, input_tokens: int = 0, output_tokens: int = 0) -> "TokenUsage":
        """Create usage backed by the local conservative estimator."""
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source="estimated",
            _input_source="estimated",
            _output_source="estimated",
        )

    @classmethod
    def from_upstream(cls, usage: object) -> "TokenUsage":
        """Create usage from counts reported by an upstream service."""
        if isinstance(usage, cls):
            return usage
        return cls().with_upstream(usage)

    def with_estimated(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> "TokenUsage":
        """Fill or refresh estimated fields, preserving upstream fields."""
        next_input = self.input_tokens
        next_output = self.output_tokens
        input_source = self._input_source or "unavailable"
        output_source = self._output_source or "unavailable"
        if input_tokens is not None and input_source != "upstream":
            next_input = _non_negative_int(input_tokens)
            input_source = "estimated"
        if output_tokens is not None and output_source != "upstream":
            next_output = _non_negative_int(output_tokens)
            output_source = "estimated"
        return self._from_fields(next_input, next_output, input_source, output_source)

    def with_upstream(self, usage: object) -> "TokenUsage":
        """Overlay valid upstream counts while retaining missing estimates."""
        if not isinstance(usage, dict):
            return self
        input_value = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_value = usage.get("completion_tokens", usage.get("output_tokens"))
        input_tokens = _parse_count(input_value)
        output_tokens = _parse_count(output_value)
        if input_tokens is None and output_tokens is None:
            return self

        next_input = self.input_tokens
        next_output = self.output_tokens
        input_source = self._input_source or "unavailable"
        output_source = self._output_source or "unavailable"
        if input_tokens is not None:
            next_input = input_tokens
            input_source = "upstream"
        if output_tokens is not None:
            next_output = output_tokens
            output_source = "upstream"
        return self._from_fields(next_input, next_output, input_source, output_source)

    @classmethod
    def _from_fields(
        cls,
        input_tokens: int,
        output_tokens: int,
        input_source: _FieldSource,
        output_source: _FieldSource,
    ) -> "TokenUsage":
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source="unavailable",
            _input_source=input_source,
            _output_source=output_source,
        )

    @staticmethod
    def _combined_source(input_source: _FieldSource, output_source: _FieldSource) -> UsageSource:
        sources = {input_source, output_source} - {"unavailable"}
        if not sources:
            return "unavailable"
        if len(sources) == 1:
            return next(iter(sources))  # type: ignore[return-value]
        return "mixed"


def _parse_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def estimate_conservative_tokens(value: object) -> int:
    """Return a high-biased token estimate for text or JSON-serializable data."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(value)

    if not text:
        return 0

    byte_tokens = math.ceil(len(text.encode("utf-8")) / CONSERVATIVE_BYTES_PER_TOKEN)
    punctuation_tokens = math.ceil(sum(char in string.punctuation for char in text) / 2)
    line_break_tokens = text.count("\n")
    return max(1, byte_tokens + punctuation_tokens + line_break_tokens + 4)


def estimate_conservative_prompt_tokens(value: object) -> int:
    """Estimate prompt tokens, including hidden protocol/template overhead."""
    return estimate_conservative_tokens(value) + PROMPT_PROTOCOL_OVERHEAD_TOKENS
