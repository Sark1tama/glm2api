"""Errors shared by the ChatGLM transport and vertical slices."""

from __future__ import annotations


class UpstreamAPIError(RuntimeError):
    """An upstream request or response could not be completed."""

    def __init__(
        self,
        status_code: int,
        message: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class QueueTimeoutError(RuntimeError):
    """The local request queue did not acquire a slot before its deadline."""


__all__ = ["QueueTimeoutError", "UpstreamAPIError"]
