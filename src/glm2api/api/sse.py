"""Shared Server-Sent Events response writing helpers."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from http import HTTPStatus
from logging import Logger


CLIENT_DISCONNECTED = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout)


class SSEWriter:
    """Write and flush UTF-8 SSE frames to an HTTP handler stream."""

    def __init__(self, wfile) -> None:
        self.wfile = wfile

    def write(self, payload: str | bytes) -> None:
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        self.wfile.write(data)
        self.wfile.flush()

    def keepalive(self) -> None:
        self.write(b": keep-alive\n\n")

    def done(self) -> None:
        self.write(b"data: [DONE]\n\n")


def start_sse_response(
    handler,
    common_headers: Callable[[], None],
    *,
    request_id: str | None = None,
) -> SSEWriter:
    """Send the common streaming headers and return a writer for the body."""
    handler.send_response(HTTPStatus.OK)
    common_headers()
    if request_id:
        handler.send_header("request-id", request_id)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()
    return SSEWriter(handler.wfile)


def write_sse_error(writer: SSEWriter, message: str, error_type: str, *, logger: Logger, path: str) -> None:
    event = {"error": {"message": message, "type": error_type}}
    try:
        writer.write(f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n")
    except CLIENT_DISCONNECTED:
        logger.warning("客户端在 SSE 错误写回前断开 path=%s", path)


__all__ = [
    "CLIENT_DISCONNECTED",
    "SSEWriter",
    "start_sse_response",
    "write_sse_error",
]
