"""ChatGLM SSE event aggregation into protocol-neutral text events."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from logging import Logger

from ..core.models import Message, TextGenerationResponse, TextStreamEvent, ToolCall, ToolCallDelta, ToolChoice
from ..core.output_budget import OutputTokenBudget, bound_output
from ..core.usage import TokenUsage, estimate_conservative_tokens
from ..infrastructure.logging import debug_dump
from ..utils.json import safe_json_dumps
from .tools.parser import StreamingToolParser, parse_tool_calls_from_text
from .translator import sanitize_tool_calls


_REMOTE_SANDBOX_TOOL_NAME = "execute_sandbox_code"


@dataclass
class _StopSequenceFilter:
    sequences: tuple[str, ...] = ()
    pending: str = ""
    matched: str | None = None
    _accepted_parts: list[str] = field(default_factory=list)

    def consume(self, text: str) -> str:
        if not text:
            return ""
        if not self.sequences:
            return text
        if self.matched is not None:
            return ""

        combined = self.pending + text
        match: tuple[int, int, str] | None = None
        for order, sequence in enumerate(self.sequences):
            index = combined.find(sequence)
            if index >= 0 and (match is None or (index, order) < match[:2]):
                match = (index, order, sequence)

        if match is not None:
            accepted = combined[: match[0]]
            self.pending = ""
            self.matched = match[2]
            self._accepted_parts.append(accepted)
            return accepted

        hold_length = max((len(sequence) for sequence in self.sequences), default=1) - 1
        if hold_length <= 0:
            accepted = combined
            self.pending = ""
        elif len(combined) <= hold_length:
            accepted = ""
            self.pending = combined
        else:
            accepted = combined[:-hold_length]
            self.pending = combined[-hold_length:]
        self._accepted_parts.append(accepted)
        return accepted

    def flush(self) -> str:
        if not self.sequences or self.matched is not None:
            return ""
        accepted = self.pending
        self.pending = ""
        self._accepted_parts.append(accepted)
        return accepted

    def text(self) -> str:
        if not self.sequences:
            return ""
        pending = "" if self.matched is not None else self.pending
        return "".join(self._accepted_parts) + pending


def effective_event_status(event: dict[str, object]) -> object:
    """Use ``code`` when an upstream envelope has no usable status value."""
    status = event.get("status")
    if status is None or (isinstance(status, str) and not status.strip()):
        return event.get("code")
    return status


def is_nonzero_status(value: object) -> bool:
    """Recognize numeric ChatGLM failure statuses, including JSON strings."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return False
        try:
            return float(normalized) != 0
        except ValueError:
            return False
    return False


@dataclass
class GLMUpstreamEventAccumulator:
    model: str
    allowed_tool_names: set[str] | None = None
    fallback_tool_url: str | None = None
    debug_enabled: bool = False
    logger: Logger | None = None
    conversation_id: str = ""
    created: int = field(default_factory=lambda: int(time.time()))
    parts_by_logic_id: dict[str, dict[str, object]] = field(default_factory=dict)
    ordered_logic_ids: list[str] = field(default_factory=list)
    last_full_text: str = ""
    last_full_reasoning: str = ""
    _part_text_sent: dict[str, int] = field(default_factory=dict)
    _part_reasoning_sent: dict[str, int] = field(default_factory=dict)
    _known_logic_ids_for_text: list[str] = field(default_factory=list)
    _known_logic_ids_for_reasoning: list[str] = field(default_factory=list)
    tool_parser: StreamingToolParser = field(default_factory=StreamingToolParser)
    emitted_role: bool = False
    _render_cache_dirty: bool = True
    _cached_full_text: str = ""
    _cached_full_reasoning: str = ""
    _cached_part_texts: dict[str, str] = field(default_factory=dict)
    _cached_part_reasonings: dict[str, str] = field(default_factory=dict)
    _part_text_accumulated: dict[str, str] = field(default_factory=dict)
    _part_reasoning_accumulated: dict[str, str] = field(default_factory=dict)
    _part_text_modes: dict[str, str] = field(default_factory=dict)
    _part_reasoning_modes: dict[str, str] = field(default_factory=dict)
    _deferred_visible_text: str = ""
    input_tokens_estimate: int = 0
    usage: TokenUsage | None = None
    tool_choice: ToolChoice | None = None
    max_output_tokens: int | None = None
    stop_sequences: tuple[str, ...] = ()
    _provider_tool_names: set[str] = field(default_factory=set)
    _output_budget: OutputTokenBudget = field(init=False, repr=False)
    _stop_filter: _StopSequenceFilter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.allowed_tool_names is None:
            self.allowed_tool_names = set()
        self.tool_parser.allowed_tool_names = self.allowed_tool_names
        initial_usage = self.usage or TokenUsage()
        if self.input_tokens_estimate:
            initial_usage = initial_usage.with_estimated(input_tokens=self.input_tokens_estimate)
        self.usage = initial_usage
        self._output_budget = OutputTokenBudget(limit=self.max_output_tokens)
        self._stop_filter = _StopSequenceFilter(
            tuple(sequence for sequence in self.stop_sequences if sequence)
        )

    @property
    def output_budget_exhausted(self) -> bool:
        """Whether a streaming caller can stop consuming the GLM response."""
        return self._output_budget.limit_reached

    @property
    def stop_sequence_matched(self) -> str | None:
        return self._stop_filter.matched

    @property
    def provider_tool_names(self) -> frozenset[str]:
        return frozenset(self._provider_tool_names)

    @property
    def remote_sandbox_used(self) -> bool:
        return _REMOTE_SANDBOX_TOOL_NAME in self._provider_tool_names

    def has_client_tool_call(self) -> bool:
        """Return whether this attempt produced a request-declared client call."""
        if self.allowed_tool_names is None:
            return False
        _, text_calls = parse_tool_calls_from_text(
            self._effective_full_text().strip(),
            allowed_tool_names=self.allowed_tool_names,
        )
        return bool(sanitize_tool_calls(text_calls, fallback_url=self.fallback_tool_url))

    def usage_snapshot(self) -> TokenUsage:
        """Return usage observed so far for a discarded upstream attempt."""
        full_text, full_reasoning = self._render_full_output()
        return self._build_usage(
            [],
            output_text=full_text,
            output_reasoning=full_reasoning,
        )

    def consume_event(self, payload: dict[str, object]) -> tuple[list[TextStreamEvent], str | None]:
        """Consume one ChatGLM upstream event into protocol-neutral events."""
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM SSE 解析事件", payload)
        self._record_upstream_usage(payload.get("usage"))
        if not self.conversation_id and payload.get("conversation_id"):
            self.conversation_id = str(payload["conversation_id"])

        payload_status = str(payload.get("status")).strip() if payload.get("status") is not None else None
        authoritative_snapshot = (payload_status or "").lower() in {"finish", "intervene", "done"}

        for part in payload.get("parts", []) if isinstance(payload.get("parts"), list) else []: # pyright: ignore[reportGeneralTypeIssues]
            if isinstance(part, dict):
                self._record_provider_tool_activity(part)
            if isinstance(part, dict) and part.get("logic_id"):
                logic_id = str(part["logic_id"])
                if logic_id not in self.parts_by_logic_id:
                    # Upstream logic IDs are opaque identifiers, not sequence
                    # numbers.  Preserve arrival order instead of sorting IDs
                    # lexicographically and reordering the answer.
                    self.ordered_logic_ids.append(logic_id)
                self.parts_by_logic_id[logic_id] = part
                self._render_cache_dirty = True
                part_text, part_reasoning = self._render_part_content(part)
                self._merge_upstream_segment(
                    self._part_text_accumulated,
                    self._part_text_modes,
                    logic_id,
                    part_text,
                    authoritative_snapshot=authoritative_snapshot,
                )
                self._merge_upstream_segment(
                    self._part_reasoning_accumulated,
                    self._part_reasoning_modes,
                    logic_id,
                    part_reasoning,
                    authoritative_snapshot=authoritative_snapshot,
                )
        text_delta, reasoning_delta = self._compute_deltas()
        self.last_full_text = self._cached_full_text
        self.last_full_reasoning = self._cached_full_reasoning

        events: list[TextStreamEvent] = []
        accepted_reasoning_delta = self._output_budget.accept_reasoning(reasoning_delta)
        if accepted_reasoning_delta:
            events.append(
                self._stream_event(
                    "reasoning_delta",
                    reasoning_content=accepted_reasoning_delta,
                )
            )

        visible_text_delta = self.tool_parser.consume(
            self._stop_filter.consume(text_delta)
        )
        if visible_text_delta:
            if self.allowed_tool_names:
                self._deferred_visible_text += visible_text_delta
            else:
                accepted_visible_text = self._output_budget.accept_text(visible_text_delta)
                if accepted_visible_text:
                    role = None
                    if not self.emitted_role:
                        role = "assistant"
                        self.emitted_role = True
                    events.append(
                        self._stream_event(
                            "text_delta",
                            role=role,
                            text=accepted_visible_text,
                        )
                    )
        debug_dump(
            self.logger or logging.getLogger("glm2api.null"),
            self.debug_enabled,
            "GLM SSE 生成内部增量事件",
            events,
        )
        return events, payload_status

    def _record_provider_tool_activity(self, part: dict[str, object]) -> None:
        meta_data = part.get("meta_data")
        observed_names: set[str] = set()

        if isinstance(meta_data, dict):
            tool_result_extra = meta_data.get("tool_result_extra")
            if isinstance(tool_result_extra, dict):
                tool_name = str(tool_result_extra.get("tool_call_name", "")).strip()
                if tool_name and tool_name != "finish":
                    observed_names.add(tool_name)

        if isinstance(part.get("content"), list):
            for content in part["content"]:
                if not isinstance(content, dict) or content.get("type") not in {"tool_calls", "tool_result"}:
                    continue
                tool_call = content.get("tool_calls")
                if isinstance(tool_call, dict):
                    tool_name = str(tool_call.get("name", "")).strip()
                    if tool_name and tool_name != "finish":
                        observed_names.add(tool_name)

        new_names = observed_names - self._provider_tool_names
        self._provider_tool_names.update(observed_names)
        if self.logger:
            for tool_name in sorted(new_names):
                self.logger.info("检测到 GLM Provider 工具活动 tool=%s", tool_name)

    def finalize(
        self,
        status: str | None,
        last_error: dict[str, object] | None = None,
    ) -> list[TextStreamEvent]:
        """Finalize into protocol-neutral stream events."""
        stop_tail = self.tool_parser.consume(self._stop_filter.flush())
        tail_text, xml_tool_calls = self.tool_parser.flush()
        all_tool_calls, xml_tool_calls = self._collect_tool_calls(xml_tool_calls)

        if self.logger:
            self.logger.info(
                "响应收尾 status=%s text_len=%s reasoning_len=%s tool_calls=%s provider_tools=%s",
                status,
                len(self._cached_full_text),
                len(self._cached_full_reasoning),
                len(xml_tool_calls),
                ",".join(sorted(self._provider_tool_names)) or "(none)",
            )

        events: list[TextStreamEvent] = []
        final_text = self._deferred_visible_text + stop_tail + tail_text
        self._deferred_visible_text = ""
        source_text = self._effective_full_text()
        if not final_text and not all_tool_calls and self.allowed_tool_names is not None:
            _, attempted_tool_calls = parse_tool_calls_from_text(
                source_text.strip(),
                allowed_tool_names=None,
            )
            unavailable_names = sorted(
                {
                    str(tool_call.get("function", {}).get("name", "")).strip()
                    for tool_call in attempted_tool_calls
                    if isinstance(tool_call.get("function"), dict)
                    and str(tool_call.get("function", {}).get("name", "")).strip()
                    not in self.allowed_tool_names
                }
            )
            if unavailable_names:
                allowed_names = ", ".join(sorted(self.allowed_tool_names)) or "(none)"
                final_text = (
                    "模型尝试调用未声明工具 "
                    + ", ".join(f"`{name}`" for name in unavailable_names)
                    + f"，已阻止。本轮只允许这些工具：{allowed_names}。"
                )
        # A completed tool call is the response payload.  This accumulator has
        # always suppressed any accompanying narration, so it must not consume
        # budget that belongs to the call.
        accepted_final_text = self._output_budget.accept_text(
            "" if all_tool_calls else final_text
        )

        intervention_text = ""
        if status == "intervene" and last_error and last_error.get("intervene_text"):
            intervention_text = str(last_error["intervene_text"])
        accepted_intervention_text = self._output_budget.accept_text(
            "\n\n" + intervention_text if intervention_text else ""
        )
        accepted_tool_calls = self._output_budget.accept_tool_calls(all_tool_calls)

        finish_reason = (
            "length"
            if self._output_budget.limit_reached
            else "tool_calls"
            if accepted_tool_calls
            else "stop"
        )
        matched_stop_sequence = (
            self.stop_sequence_matched
            if finish_reason == "stop"
            else None
        )
        self._validate_tool_choice(
            all_tool_calls,
            source_text=source_text,
            allow_missing=finish_reason == "length",
        )

        if accepted_final_text and not accepted_tool_calls:
            role = None
            if not self.emitted_role:
                role = "assistant"
                self.emitted_role = True
            events.append(
                self._stream_event(
                    "text_delta",
                    role=role,
                    text=accepted_final_text,
                )
            )

        if accepted_intervention_text:
            events.append(
                self._stream_event(
                    "text_delta",
                    text=accepted_intervention_text,
                )
            )

        if accepted_tool_calls:
            if not self.emitted_role:
                events.append(
                    self._stream_event(
                        "role",
                        role="assistant",
                    )
                )
                self.emitted_role = True
            for tool_call in accepted_tool_calls:
                function = tool_call.get("function", {})
                if not isinstance(function, dict):
                    continue
                tool_name = str(function.get("name", "")).strip()
                if not tool_name:
                    continue
                raw_arguments = function.get("arguments", "")
                arguments = raw_arguments if isinstance(raw_arguments, str) else safe_json_dumps(raw_arguments)
                events.append(
                    self._stream_event(
                        "tool_call_delta",
                        tool_call=ToolCallDelta(
                            index=int(tool_call.get("index", 0)),
                            id=str(tool_call.get("id", "")) or None,
                            name=tool_name,
                            arguments=arguments,
                        ),
                    )
                )

        usage = self._build_usage(
            list(accepted_tool_calls),
            output_text=self._output_budget.text,
            output_reasoning=self._output_budget.reasoning,
        )
        events.append(
            self._stream_event(
                "finish",
                finish_reason=finish_reason,
                stop_sequence=matched_stop_sequence,
                usage=usage,
            )
        )
        events.append(self._stream_event("done"))
        return events

    def build_response(self) -> TextGenerationResponse:
        _, full_reasoning = self._render_full_output()
        if not full_reasoning and self.last_full_reasoning:
            full_reasoning = self.last_full_reasoning
        self._stop_filter.flush()
        full_text = self._effective_full_text()
        clean_content, xml_tool_calls = parse_tool_calls_from_text(
            full_text.strip(),
            allowed_tool_names=self.allowed_tool_names,
        )
        all_tool_calls, _ = self._collect_tool_calls(xml_tool_calls)

        final_content = "" if all_tool_calls else clean_content.strip()
        bounded = bound_output(
            self.max_output_tokens,
            reasoning=full_reasoning,
            text=final_content,
            tool_calls=all_tool_calls,
        )
        finish_reason = (
            "length"
            if bounded.limit_reached
            else "tool_calls"
            if bounded.tool_calls
            else "stop"
        )
        matched_stop_sequence = (
            self.stop_sequence_matched
            if finish_reason == "stop"
            else None
        )
        self._validate_tool_choice(
            all_tool_calls,
            source_text=full_text,
            allow_missing=finish_reason == "length",
        )
        internal_tool_calls: list[ToolCall] = []
        for index, item in enumerate(bounded.tool_calls):
            function = item.get("function") if isinstance(item, dict) else None
            if not isinstance(function, dict):
                continue
            tool_name = str(function.get("name", "")).strip()
            if not tool_name:
                continue
            raw_arguments = function.get("arguments", "{}")
            arguments = raw_arguments if isinstance(raw_arguments, str) else safe_json_dumps(raw_arguments)
            internal_tool_calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_repaired_{index}"),
                    name=tool_name,
                    arguments=arguments,
                )
            )
        message = Message(
            role="assistant",
            content=None if internal_tool_calls or not bounded.text else bounded.text,
            reasoning_content=bounded.reasoning or None,
            tool_calls=tuple(internal_tool_calls),
        )
        usage = self._build_usage(
            list(bounded.tool_calls),
            output_text=bounded.text,
            output_reasoning=bounded.reasoning,
        )
        response = TextGenerationResponse(
            response_id=self.conversation_id,
            model=self.model,
            created=self.created,
            message=message,
            finish_reason=finish_reason,
            usage=usage,
            stop_sequence=matched_stop_sequence,
        )
        if self.logger:
            self.logger.info(
                "非流式响应构建完成 model=%s text_len=%s reasoning_len=%s tool_calls=%s",
                self.model,
                len(bounded.text),
                len(bounded.reasoning),
                len(bounded.tool_calls),
            )
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM 非流式最终响应", response)
        return response

    def _effective_full_text(self) -> str:
        if self._stop_filter.sequences:
            return self._stop_filter.text()
        full_text, _ = self._render_full_output()
        return full_text or self.last_full_text

    def _collect_tool_calls(
        self,
        parsed_tool_calls: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Normalize executable calls parsed from assistant text."""
        xml_tool_calls = sanitize_tool_calls(
            parsed_tool_calls,
            fallback_url=self.fallback_tool_url,
        )

        all_tool_calls: list[dict[str, object]] = []
        for tool_call in xml_tool_calls:
            normalized_call = dict(tool_call)
            normalized_call["index"] = len(all_tool_calls)
            all_tool_calls.append(normalized_call)

        return all_tool_calls, xml_tool_calls

    def _validate_tool_choice(
        self,
        all_tool_calls: list[dict[str, object]],
        *,
        source_text: str,
        allow_missing: bool = False,
    ) -> None:
        choice = self.tool_choice
        if choice is None or choice.mode == "auto":
            return

        attempted_names: list[str] = []
        _, attempted_calls = parse_tool_calls_from_text(source_text.strip(), allowed_tool_names=None)
        attempted_names.extend(
            str(call.get("function", {}).get("name", "")).strip()
            for call in attempted_calls
            if isinstance(call.get("function"), dict)
        )
        actual_names = [
            str(call.get("function", {}).get("name", "")).strip()
            for call in all_tool_calls
            if isinstance(call.get("function"), dict)
        ]
        if choice.mode == "none":
            if actual_names or attempted_names:
                raise ValueError("tool_choice=none 时模型输出了工具调用")
            return

        if choice.mode == "function":
            unexpected_names = {
                name
                for name in actual_names + attempted_names
                if name and name != choice.name
            }
            if unexpected_names:
                raise ValueError(
                    f"tool_choice 指定 {choice.name}，但模型输出了其他工具: "
                    f"{', '.join(sorted(unexpected_names))}"
                )
        if not actual_names:
            if allow_missing and not any(attempted_names):
                return
            raise ValueError(f"tool_choice={choice.mode} 但模型未输出工具调用")

    def _record_upstream_usage(self, usage: object) -> None:
        self.usage = (self.usage or TokenUsage()).with_upstream(usage)

    def _build_usage(
        self,
        all_tool_calls: list[dict[str, object]],
        extra_output_text: str = "",
        *,
        output_text: str | None = None,
        output_reasoning: str | None = None,
    ) -> TokenUsage:
        usage = (self.usage or TokenUsage()).with_estimated(
            input_tokens=self.input_tokens_estimate if self.input_tokens_estimate > 0 else None,
            output_tokens=self._estimate_output_tokens(
                all_tool_calls,
                extra_output_text=extra_output_text,
                output_text=output_text,
                output_reasoning=output_reasoning,
            ),
        )
        self.usage = usage
        return usage

    def _estimate_output_tokens(
        self,
        all_tool_calls: list[dict[str, object]],
        extra_output_text: str = "",
        *,
        output_text: str | None = None,
        output_reasoning: str | None = None,
    ) -> int:
        self._render_full_output()
        output_parts = [
            self._cached_full_reasoning if output_reasoning is None else output_reasoning,
            self._cached_full_text if output_text is None else output_text,
        ]
        if all_tool_calls:
            output_parts.append(safe_json_dumps(all_tool_calls))
        if extra_output_text:
            output_parts.append(extra_output_text)
        output_text = "\n".join(part for part in output_parts if part)
        return estimate_conservative_tokens(output_text)

    def _compute_deltas(self) -> tuple[str, str]:
        self._render_full_output()
        text_delta_parts: list[str] = []
        reasoning_delta_parts: list[str] = []

        for logic_id in self.ordered_logic_ids:
            rendered_text = self._cached_part_texts.get(logic_id, "")
            rendered_reasoning = self._cached_part_reasonings.get(logic_id, "")

            if rendered_text:
                prev_len = self._part_text_sent.get(logic_id, 0)
                is_new = logic_id not in self._known_logic_ids_for_text
                if is_new:
                    self._known_logic_ids_for_text.append(logic_id)
                    if text_delta_parts or self._part_text_sent:
                        text_delta_parts.append("\n\n")
                    text_delta_parts.append(rendered_text)
                elif len(rendered_text) > prev_len:
                    text_delta_parts.append(rendered_text[prev_len:])
                self._part_text_sent[logic_id] = len(rendered_text)

            if rendered_reasoning:
                prev_len = self._part_reasoning_sent.get(logic_id, 0)
                is_new = logic_id not in self._known_logic_ids_for_reasoning
                if is_new:
                    self._known_logic_ids_for_reasoning.append(logic_id)
                    if reasoning_delta_parts or self._part_reasoning_sent:
                        reasoning_delta_parts.append("\n\n")
                    reasoning_delta_parts.append(rendered_reasoning)
                elif len(rendered_reasoning) > prev_len:
                    reasoning_delta_parts.append(rendered_reasoning[prev_len:])
                self._part_reasoning_sent[logic_id] = len(rendered_reasoning)

        return "".join(text_delta_parts), "".join(reasoning_delta_parts)

    @staticmethod
    def _merge_upstream_segment(
        accumulated: dict[str, str],
        modes: dict[str, str],
        logic_id: str,
        segment: str,
        *,
        authoritative_snapshot: bool,
    ) -> None:
        """Merge ChatGLM's delta events and final full-text snapshot.

        The web SSE normally emits short text deltas while a response is in
        progress, then repeats the complete part text in the ``finish``
        event.  Older responses may instead send cumulative snapshots, so
        detect that shape per logic part and support both forms.
        """
        if not segment:
            return

        previous = accumulated.get(logic_id, "")
        if authoritative_snapshot:
            accumulated[logic_id] = segment
            modes[logic_id] = "snapshot"
            return

        if not previous:
            accumulated[logic_id] = segment
            return

        mode = modes.get(logic_id)
        if mode == "snapshot" or segment.startswith(previous):
            accumulated[logic_id] = segment
            modes[logic_id] = "snapshot"
            return

        accumulated[logic_id] = previous + segment
        modes[logic_id] = "delta"

    @staticmethod
    def _render_part_content(part: dict[str, object]) -> tuple[str, str]:
        content_items = part.get("content", [])
        if not isinstance(content_items, list):
            return "", ""

        part_text: list[str] = []
        part_reasoning: list[str] = []
        for content in content_items:
            if not isinstance(content, dict):
                continue
            item_type = content.get("type")
            if item_type == "text":
                part_text.append(str(content.get("text", "")))
            elif item_type == "think":
                part_reasoning.append(str(content.get("think", "")))
            elif item_type == "code":
                part_text.append(f"```python\n{content.get('code', '')}\n```")
            elif item_type == "execution_output":
                part_text.append(str(content.get("content", "")))
            elif item_type == "image":
                images = content.get("image", [])
                if isinstance(images, list):
                    for image in images:
                        if isinstance(image, dict) and image.get("image_url"):
                            part_text.append(f"![image]({image['image_url']})")

        return "\n".join(filter(None, part_text)), "\n".join(filter(None, part_reasoning))

    def _render_full_output(self) -> tuple[str, str]:
        if not self._render_cache_dirty:
            return self._cached_full_text, self._cached_full_reasoning

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        self._cached_part_texts.clear()
        self._cached_part_reasonings.clear()
        for logic_id in self.ordered_logic_ids:
            part = self.parts_by_logic_id.get(logic_id)
            if not isinstance(part, dict):
                continue
            fallback_text, fallback_reasoning = self._render_part_content(part)
            rendered_text = self._part_text_accumulated.get(logic_id, fallback_text).strip()
            rendered_reasoning = self._part_reasoning_accumulated.get(logic_id, fallback_reasoning).strip()
            if rendered_text:
                text_parts.append(rendered_text)
                self._cached_part_texts[logic_id] = rendered_text
            if rendered_reasoning:
                reasoning_parts.append(rendered_reasoning)
                self._cached_part_reasonings[logic_id] = rendered_reasoning

        self._cached_full_text = "\n\n".join(text_parts)
        self._cached_full_reasoning = "\n\n".join(reasoning_parts)
        self._render_cache_dirty = False
        return self._cached_full_text, self._cached_full_reasoning

    def _stream_event(self, kind: str, **kwargs: object) -> TextStreamEvent:
        return TextStreamEvent(
            kind=kind,
            response_id=self.conversation_id,
            model=self.model,
            created=self.created,
            **kwargs,
        )
