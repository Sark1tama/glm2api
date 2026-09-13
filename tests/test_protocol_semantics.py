import json

import pytest

from glm2api.api.adapters.anthropic.messages import anthropic_messages_to_internal
from glm2api.api.adapters.openai.chat_completions import openai_chat_completions_to_internal
from glm2api.api.adapters.openai.responses import (
    OpenAIResponsesStreamAccumulator,
    internal_to_openai_responses_response,
    openai_responses_to_internal,
)
from glm2api.core.models import ToolChoice
from glm2api.glm.events import GLMUpstreamEventAccumulator
from glm2api.glm.tools.dsml import build_tool_call_instructions


def tool_block(names=("a",)):
    return "<|DSML|tool_calls>" + "".join(
        f'<|DSML|invoke name="{name}"><|DSML|parameter name="command">echo STOP</|DSML|parameter></|DSML|invoke>'
        for name in names
    ) + "</|DSML|tool_calls>"


def consume(accumulator, text):
    # Cumulative upstream snapshots, including splits within STOP and DSML tags.
    events = []
    for length in range(1, len(text) + 1):
        delta, _ = accumulator.consume_event({"parts": [{
            "logic_id": "one", "content": [{"type": "text", "text": text[:length]}],
        }]})
        events.extend(delta)
        if accumulator.stop_sequence_matched or accumulator.has_client_tool_call():
            break
    return events


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("prefix", ["Ready. ", "Ready. STOP ignored "])
def test_stop_only_matches_visible_text(streaming, prefix):
    accumulator = GLMUpstreamEventAccumulator(
        model="test", allowed_tool_names={"a"}, stop_sequences=("STOP",),
        tool_choice=ToolChoice(mode="required"),
    )
    events = consume(accumulator, prefix + tool_block())
    stopped = "STOP" in prefix
    if streaming:
        events.extend(accumulator.finalize("finish"))
        calls = [event.tool_call for event in events if event.kind == "tool_call_delta"]
        text = "".join(event.text for event in events)
        finish = next(event for event in events if event.kind == "finish")
        reason = finish.finish_reason
    else:
        response = accumulator.build_response()
        calls = response.message.tool_calls
        text = response.message.content
        reason = response.finish_reason
    assert text.strip() == "Ready."
    assert len(calls) == (0 if stopped else 1)
    assert reason == ("stop" if stopped else "tool_calls")
    if calls:
        assert json.loads(calls[0].arguments)["command"] == "echo STOP"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_parallel_constraint_is_enforced(streaming, parallel):
    accumulator = GLMUpstreamEventAccumulator(
        model="test", allowed_tool_names={"a", "b"},
        tool_choice=ToolChoice(mode="auto", parallel_tool_calls=parallel),
    )
    consume(accumulator, tool_block(("a", "b")))
    if streaming:
        calls = [e.tool_call for e in accumulator.finalize("finish") if e.kind == "tool_call_delta"]
    else:
        calls = accumulator.build_response().message.tool_calls
    assert [call.name for call in calls] == (["a", "b"] if parallel else ["a"])


@pytest.mark.parametrize("value", [False, True, "false"])
def test_parallel_parameter_mapping(value):
    conversions = [
        (openai_chat_completions_to_internal, {"messages": [], "parallel_tool_calls": value}),
        (openai_responses_to_internal, {"input": "test", "parallel_tool_calls": value}),
        (anthropic_messages_to_internal, {"messages": [], "tool_choice": {
            "type": "auto", "disable_parallel_tool_use": not value if isinstance(value, bool) else value,
        }}),
    ]
    for convert, payload in conversions:
        if not isinstance(value, bool):
            with pytest.raises(ValueError):
                convert(payload)
        else:
            request = convert(payload)
            assert (request.tool_choice.parallel_tool_calls if request.tool_choice else True) == value


def test_responses_rejects_unresolved_previous_response():
    with pytest.raises(ValueError, match="完整对话历史"):
        openai_responses_to_internal({"input": "continue", "previous_response_id": "resp_missing"})
    assert openai_responses_to_internal({"input": "hello", "previous_response_id": None}).messages


@pytest.mark.parametrize("split", [False, True])
def test_stop_with_trailing_space_before_tool_matches_in_both_modes(split):
    text = "helloEND " + tool_block()
    results = []
    for streaming in (False, True):
        accumulator = GLMUpstreamEventAccumulator(
            model="test", allowed_tool_names={"a"}, stop_sequences=("END ",),
            tool_choice=ToolChoice(mode="required"),
        )
        if split:
            events = consume(accumulator, text)
        else:
            events, _ = accumulator.consume_event({"parts": [{
                "logic_id": "one", "content": [{"type": "text", "text": text}],
            }]})
        if streaming:
            events.extend(accumulator.finalize("finish"))
            finish = next(e for e in events if e.kind == "finish")
            results.append(("".join(e.text for e in events), finish.stop_sequence))
            assert not any(e.kind == "tool_call_delta" for e in events)
        else:
            response = accumulator.build_response()
            results.append((response.message.content, response.stop_sequence))
            assert not response.message.tool_calls
    assert results == [("hello", "END "), ("hello", "END ")]


@pytest.mark.parametrize("parallel", [False, True])
def test_parallel_prompt_has_one_consistent_policy(parallel):
    prompt = build_tool_call_instructions(["a", "b"], {"mode": "auto", "parallel_tool_calls": parallel})
    assert ("Put multiple DSML invokes" in prompt) == parallel
    assert ("Emit at most one client-side tool call" in prompt) == (not parallel)


def test_responses_reports_request_tool_metadata_in_both_modes():
    request = openai_responses_to_internal({
        "input": "test", "tools": [{"type": "function", "name": "a", "parameters": {"type": "object"}}],
        "tool_choice": {"type": "function", "name": "a"}, "parallel_tool_calls": False,
    })
    result = GLMUpstreamEventAccumulator(model="test").build_response()
    response = internal_to_openai_responses_response(result, "test", request=request)
    stream = OpenAIResponsesStreamAccumulator(model="test", request=request)
    payload = json.loads(stream.start_response()[0].split("data: ")[1])["response"]
    for output in (response, payload):
        assert output["parallel_tool_calls"] is False
        assert output["tool_choice"] == {"type": "function", "name": "a"}
        assert output["tools"][0]["name"] == "a"
        assert output["tools"][0]["parameters"] == {"type": "object"}
