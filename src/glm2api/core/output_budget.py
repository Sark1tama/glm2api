"""Conservative local output-token budgeting for the GLM web adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from .usage import estimate_conservative_tokens


def _serialize_tool_calls(tool_calls: Sequence[dict[str, object]]) -> str:
    return json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class BoundedOutput:
    """Output accepted by a local token budget."""

    reasoning: str
    text: str
    tool_calls: tuple[dict[str, object], ...]
    truncated: bool
    limit_reached: bool
    output_tokens: int


@dataclass(slots=True)
class OutputTokenBudget:
    """Apply a conservative output-token limit without splitting tool JSON."""

    limit: int | None = None
    reasoning: str = ""
    text: str = ""
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    truncated: bool = False

    def __post_init__(self) -> None:
        if self.limit is not None and (
            isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 0
        ):
            raise ValueError("output token limit must be a non-negative integer")

    @property
    def output_tokens(self) -> int:
        return self._estimate(self.reasoning, self.text, self.tool_calls)

    @property
    def limit_reached(self) -> bool:
        """Whether the configured budget requires a length-limited result."""
        return self.limit == 0 or self.truncated

    def accept_reasoning(self, value: str) -> str:
        accepted = self._accept_text(value, field="reasoning")
        self.reasoning += accepted
        return accepted

    def accept_text(self, value: str) -> str:
        accepted = self._accept_text(value, field="text")
        self.text += accepted
        return accepted

    def accept_tool_calls(self, values: Sequence[dict[str, object]]) -> tuple[dict[str, object], ...]:
        """Accept complete calls only; an oversized call is not partially emitted."""
        if self.truncated:
            return ()
        accepted: list[dict[str, object]] = []
        for value in values:
            candidate = self.tool_calls + accepted + [value]
            if self.limit is None or self._estimate(self.reasoning, self.text, candidate) <= self.limit:
                accepted.append(value)
                continue
            self.truncated = True
            break
        self.tool_calls.extend(accepted)
        return tuple(accepted)

    def snapshot(self) -> BoundedOutput:
        return BoundedOutput(
            reasoning=self.reasoning,
            text=self.text,
            tool_calls=tuple(self.tool_calls),
            truncated=self.truncated,
            limit_reached=self.limit_reached,
            output_tokens=self.output_tokens,
        )

    def _accept_text(self, value: str, *, field: str) -> str:
        if not value or self.truncated:
            return ""
        if self.limit is None:
            return value

        current_reasoning = self.reasoning
        current_text = self.text
        if field == "reasoning":
            candidate_reasoning = current_reasoning + value
            candidate_text = current_text
        else:
            candidate_reasoning = current_reasoning
            candidate_text = current_text + value

        if self._estimate(candidate_reasoning, candidate_text, self.tool_calls) <= self.limit:
            return value

        low = 0
        high = len(value)
        while low < high:
            middle = (low + high + 1) // 2
            prefix = value[:middle]
            if field == "reasoning":
                fits = self._estimate(current_reasoning + prefix, current_text, self.tool_calls) <= self.limit
            else:
                fits = self._estimate(current_reasoning, current_text + prefix, self.tool_calls) <= self.limit
            if fits:
                low = middle
            else:
                high = middle - 1

        self.truncated = True
        return value[:low]

    @staticmethod
    def _estimate(
        reasoning: str,
        text: str,
        tool_calls: Sequence[dict[str, object]],
    ) -> int:
        parts = [part for part in (reasoning, text) if part]
        if tool_calls:
            parts.append(_serialize_tool_calls(tool_calls))
        return estimate_conservative_tokens("\n".join(parts))


def bound_output(
    limit: int | None,
    *,
    reasoning: str,
    text: str,
    tool_calls: Sequence[dict[str, object]] = (),
) -> BoundedOutput:
    """Bound a complete output in protocol-neutral order."""
    budget = OutputTokenBudget(limit=limit)
    budget.accept_reasoning(reasoning)
    budget.accept_text(text)
    budget.accept_tool_calls(tool_calls)
    return budget.snapshot()


__all__ = ["BoundedOutput", "OutputTokenBudget", "bound_output"]
