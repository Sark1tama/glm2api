from http import HTTPStatus
from types import SimpleNamespace

from glm2api.api.errors import anthropic_error_type, build_error_payload, safe_http_status
from glm2api.api.sse import SSEWriter, start_sse_response
from glm2api.config import ConfigError, load_config
from glm2api.infrastructure.logging import debug_dump


def test_http_error_mapping_uses_safe_status_and_protocol_types():
    assert safe_http_status(400, HTTPStatus.BAD_GATEWAY) is HTTPStatus.BAD_REQUEST
    assert safe_http_status(999, HTTPStatus.BAD_GATEWAY) is HTTPStatus.BAD_GATEWAY
    assert anthropic_error_type(HTTPStatus.TOO_MANY_REQUESTS, "upstream_error") == "rate_limit_error"
    assert anthropic_error_type(HTTPStatus.BAD_GATEWAY, "queue_timeout") == "overloaded_error"
    assert build_error_payload("bad request", "invalid_request") == {
        "error": {"message": "bad request", "type": "invalid_request"}
    }


def test_sse_writer_encodes_and_flushes_frames():
    writes: list[bytes] = []
    flushes: list[bool] = []
    writer = SSEWriter(SimpleNamespace(write=writes.append, flush=lambda: flushes.append(True)))

    writer.write("data: hello\n\n")
    writer.keepalive()
    writer.done()

    assert writes == [b"data: hello\n\n", b": keep-alive\n\n", b"data: [DONE]\n\n"]
    assert len(flushes) == 3


def test_start_sse_response_sets_common_stream_headers():
    calls: list[tuple[str, object]] = []
    handler = SimpleNamespace(
        wfile=SimpleNamespace(write=lambda _data: None, flush=lambda: None),
        send_response=lambda status: calls.append(("status", status)),
        send_header=lambda name, value: calls.append((name, value)),
        end_headers=lambda: calls.append(("end", None)),
    )

    start_sse_response(handler, lambda: calls.append(("common", None)), request_id="req_1")

    assert calls[0] == ("status", HTTPStatus.OK)
    assert ("common", None) in calls
    assert ("request-id", "req_1") in calls
    assert ("Content-Type", "text/event-stream; charset=utf-8") in calls
    assert calls[-1] == ("end", None)


def test_debug_dump_redacts_credentials():
    messages: list[str] = []

    class Logger:
        def debug(self, template, *args):
            messages.append(template % args)

    debug_dump(
        Logger(),
        True,
        "headers",
        {"Authorization": "Bearer secret", "nested": {"x-api-key": "api-secret"}},
    )

    assert "secret" not in messages[0]
    assert "[REDACTED]" in messages[0]

    debug_dump(Logger(), True, "body token=title-secret", b'{"access_token":"body-secret","ok":true}')
    assert "body-secret" not in messages[1]
    assert "title-secret" not in messages[1]


def test_config_loads_and_validates_request_body_limit(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "GLM_USE_GUEST_REFRESH_TOKEN=true\nMAX_REQUEST_BODY_BYTES=123\n",
        encoding="utf-8",
    )

    config = load_config(str(env_path))
    assert config.max_request_body_bytes == 123

    env_path.write_text(
        "GLM_USE_GUEST_REFRESH_TOKEN=true\nMAX_REQUEST_BODY_BYTES=0\n",
        encoding="utf-8",
    )
    try:
        load_config(str(env_path))
    except ConfigError as exc:
        assert "请求体上限" in str(exc)
    else:
        raise AssertionError("zero request body limit must be rejected")
