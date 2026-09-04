from __future__ import annotations

from types import SimpleNamespace

import pytest

from glm2api.core.models import Message, TextGenerationRequest, ToolChoice, ToolDefinition
from glm2api.core.usage import TokenUsage
from glm2api.glm.client import GLMWebClient, QueueLease
from glm2api.glm.errors import UpstreamAPIError


class FakeResponse:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def close(self):
        self.closed = True


def _private_tool_event(conversation_id: str = "bad"):
    return {
        "conversation_id": conversation_id,
        "parts": [
            {
                "logic_id": "private",
                "content": [{"type": "think", "think": "remote result"}],
                "meta_data": {
                    "tool_result_extra": {"tool_call_name": "execute_sandbox_code"}
                },
            }
        ],
    }


def _client_for_attempts(attempt_events):
    attempts = [FakeResponse(events) for events in attempt_events]
    opened_requests = []
    deleted_conversations = []
    released_tickets = []
    warnings = []

    client = GLMWebClient.__new__(GLMWebClient)
    client.config = SimpleNamespace(blocked_tool_names=[], debug_dump_all=False)
    client.logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda message, *args, **kwargs: warnings.append(message % args),
    )
    client.request_queue = SimpleNamespace(
        acquire=lambda name: QueueLease(
            ticket=0,
            release_callback=lambda ticket: released_tickets.append(ticket),
        )
    )
    client.get_preferred_account_index = lambda ticket: 0

    def open_stream(request, preferred_account_index=None):
        opened_requests.append(request)
        usage = TokenUsage.estimated(input_tokens=10)
        request.usage = usage
        return attempts[len(opened_requests) - 1], "assistant", usage

    client._open_chat_stream = open_stream
    client.iter_sse_events = lambda response, require_done=False: iter(response.events)
    client.delete_conversation = (
        lambda conversation_id, assistant_id=None: deleted_conversations.append(conversation_id)
    )
    return client, attempts, opened_requests, deleted_conversations, released_tickets, warnings


def _request(stream: bool = False, tool_choice: ToolChoice | None = None):
    return TextGenerationRequest(
        model="glm-test",
        messages=(Message(role="user", content="检查本机配置"),),
        stream=stream,
        tool_choice=tool_choice,
        tools=(
            ToolDefinition(
                name="terminal",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            ),
        ),
    )


def _successful_tool_event():
    return {
        "conversation_id": "good",
        "status": "finish",
        "parts": [
            {
                "logic_id": "client-tool",
                "status": "finish",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '<|DSML|tool_calls><|DSML|invoke name="terminal">'
                            '<|DSML|parameter name="command"><![CDATA[uname -a]]>'
                            '</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'
                        ),
                    }
                ],
            }
        ],
    }


def test_non_streaming_private_tool_result_is_discarded_and_retried_once():
    client, responses, opened, deleted, released, warnings = _client_for_attempts(
        [[_private_tool_event()], [_successful_tool_event()]]
    )
    request = _request(tool_choice=ToolChoice(mode="required"))

    result, conversation_id = client.chat_completion(request)

    assert conversation_id == "good"
    assert [call.name for call in result.message.tool_calls] == ["terminal"]
    assert result.usage.input_tokens == 20
    assert len(opened) == 2
    assert opened[0] is request
    assert opened[1].tool_choice is not None
    assert opened[1].tool_choice.mode == "required"
    assert "# CLIENT TOOL RETRY" in str(opened[1].messages[-1].content)
    assert all(response.closed for response in responses)
    assert deleted == ["bad", "good"]
    assert released == [0]
    assert len(warnings) == 1


def test_streaming_retry_does_not_emit_discarded_attempt_events():
    client, responses, opened, deleted, released, _ = _client_for_attempts(
        [[_private_tool_event()], [_successful_tool_event()]]
    )
    request = _request(stream=True, tool_choice=ToolChoice(mode="required"))

    events = list(client.stream_chat_completion(request))

    assert not any(event.reasoning_content == "remote result" for event in events)
    assert [event.tool_call.name for event in events if event.kind == "tool_call_delta"] == ["terminal"]
    finish = next(event for event in events if event.kind == "finish")
    assert finish.usage is not None
    assert finish.usage.input_tokens == 20
    assert request.usage == finish.usage
    assert len(opened) == 2
    assert all(response.closed for response in responses)
    assert deleted == ["bad", "good"]
    assert released == [0]


def test_private_tool_retry_fails_closed_after_second_contamination():
    client, responses, opened, deleted, released, _ = _client_for_attempts(
        [[_private_tool_event("bad-1")], [_private_tool_event("bad-2")]]
    )

    with pytest.raises(UpstreamAPIError, match="结果不属于客户端环境") as raised:
        client.chat_completion(_request(tool_choice=ToolChoice(mode="required")))

    assert raised.value.status_code == 502
    assert len(opened) == 2
    assert all(response.closed for response in responses)
    assert deleted == ["bad-1", "bad-2"]
    assert released == [0]


def test_provider_web_tool_is_not_treated_as_client_tool_or_contamination():
    provider_event = {
        "conversation_id": "web",
        "parts": [
            {
                "logic_id": "provider-web",
                "content": [
                    {
                        "type": "tool_calls",
                        "tool_calls": {
                            "id": "call_web",
                            "name": "open_url",
                            "arguments": '{"urls":["https://example.com"]}',
                        },
                    }
                ],
                "meta_data": {"show_type": "mc_tool_call2"},
            }
        ],
    }
    final_event = {
        "conversation_id": "web",
        "status": "finish",
        "parts": [
            {
                "logic_id": "answer",
                "content": [{"type": "text", "text": "网页结果"}],
            }
        ],
    }
    client, responses, opened, deleted, released, _ = _client_for_attempts(
        [[provider_event, final_event]]
    )

    result, conversation_id = client.chat_completion(_request())

    assert conversation_id == "web"
    assert result.message.content == "网页结果"
    assert result.message.tool_calls == ()
    assert len(opened) == 1
    assert responses[0].closed is True
    assert deleted == ["web"]
    assert released == [0]


def test_remote_sandbox_is_allowed_for_auto_tool_choice():
    final_event = {
        "conversation_id": "calc",
        "status": "finish",
        "parts": [
            {
                "logic_id": "answer",
                "content": [{"type": "text", "text": "42"}],
            }
        ],
    }
    client, responses, opened, deleted, released, warnings = _client_for_attempts(
        [[_private_tool_event("calc"), final_event]]
    )

    result, conversation_id = client.chat_completion(_request())

    assert conversation_id == "calc"
    assert result.message.content == "42"
    assert len(opened) == 1
    assert responses[0].closed is True
    assert deleted == ["calc"]
    assert released == [0]
    assert warnings == []


def test_remote_sandbox_does_not_retry_when_attempt_also_returns_client_tool():
    client, responses, opened, deleted, released, _ = _client_for_attempts(
        [[_private_tool_event("mixed"), _successful_tool_event()]]
    )

    result, conversation_id = client.chat_completion(_request())

    assert conversation_id == "mixed"
    assert [call.name for call in result.message.tool_calls] == ["terminal"]
    assert len(opened) == 1
    assert responses[0].closed is True
    assert deleted == ["mixed"]
    assert released == [0]


def test_non_streaming_stop_sequence_ends_upstream_consumption_early():
    stop_event = {
        "conversation_id": "stopped",
        "status": "init",
        "parts": [
            {
                "logic_id": "answer",
                "content": [{"type": "text", "text": "helloENDignored"}],
            }
        ],
    }
    upstream_error = {"status": "error", "last_error": {"message": "must not be consumed"}}
    client, responses, opened, deleted, released, _ = _client_for_attempts(
        [[stop_event, upstream_error]]
    )
    request = _request()
    request.stop = "END"

    result, _ = client.chat_completion(request)

    assert result.message.content == "hello"
    assert result.stop_sequence == "END"
    assert len(opened) == 1
    assert responses[0].closed is True
    assert deleted == ["stopped"]
    assert released == [0]


def test_streaming_stop_sequence_ends_upstream_consumption_early():
    stop_event = {
        "conversation_id": "stopped",
        "status": "init",
        "parts": [
            {
                "logic_id": "answer",
                "content": [{"type": "text", "text": "helloENDignored"}],
            }
        ],
    }
    upstream_error = {"status": "error", "last_error": {"message": "must not be consumed"}}
    client, responses, opened, deleted, released, _ = _client_for_attempts(
        [[stop_event, upstream_error]]
    )
    request = _request(stream=True)
    request.stop = "END"

    events = list(client.stream_chat_completion(request))

    assert "".join(event.text for event in events if event.kind == "text_delta") == "hello"
    finish = next(event for event in events if event.kind == "finish")
    assert finish.finish_reason == "stop"
    assert finish.stop_sequence == "END"
    assert len(opened) == 1
    assert responses[0].closed is True
    assert deleted == ["stopped"]
    assert released == [0]
