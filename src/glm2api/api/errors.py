"""HTTP error status and payload mapping for the public API boundary."""

from __future__ import annotations

from http import HTTPStatus


_ANTHROPIC_ERROR_TYPES = frozenset(
    {
        "api_error",
        "authentication_error",
        "billing_error",
        "gateway_timeout",
        "invalid_request_error",
        "not_found_error",
        "overloaded_error",
        "permission_error",
        "rate_limit_error",
        "request_too_large",
    }
)


def safe_http_status(value: int, fallback: HTTPStatus) -> HTTPStatus:
    """Convert an upstream status to HTTPStatus without leaking invalid codes."""
    try:
        return HTTPStatus(value)
    except (TypeError, ValueError):
        return fallback


def anthropic_error_type(status: HTTPStatus, error_type: str) -> str:
    """Map internal errors to Anthropic's documented error type names."""
    if error_type in _ANTHROPIC_ERROR_TYPES:
        return error_type
    if error_type == "queue_timeout":
        return "overloaded_error"

    status_code = int(status)
    if status_code == HTTPStatus.UNAUTHORIZED:
        return "authentication_error"
    if status_code == HTTPStatus.TOO_MANY_REQUESTS:
        return "rate_limit_error"
    if status_code == 529:
        return "overloaded_error"
    if status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
        return "request_too_large"
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "api_error"


def build_anthropic_error_payload(error_type: str, message: str, request_id: str) -> dict[str, object]:
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
        "request_id": request_id,
    }


def build_error_payload(
    message: str,
    error_type: str,
    details: object | None = None,
) -> dict[str, object]:
    error: dict[str, object] = {"message": message, "type": error_type}
    if details is not None:
        error["details"] = details
    return {"error": error}


__all__ = [
    "anthropic_error_type",
    "build_anthropic_error_payload",
    "build_error_payload",
    "safe_http_status",
]
