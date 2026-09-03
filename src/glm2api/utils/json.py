"""JSON helpers shared by protocol and provider boundaries."""

from __future__ import annotations

import json


def safe_json_dumps(payload: object) -> str:
    """Serialize JSON deterministically without escaping non-ASCII text."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = ["safe_json_dumps"]
