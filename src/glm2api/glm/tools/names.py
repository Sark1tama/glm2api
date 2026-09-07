"""Names used to isolate client tools from ChatGLM provider tools."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from ...core.models import ToolChoice, ToolDefinition


CLIENT_TOOL_NAME_PREFIX = "client__"


@dataclass(frozen=True, slots=True)
class ClientToolNameMap:
    """Map public client tool names to an upstream-only namespace."""

    original_to_alias: dict[str, str]
    alias_to_original: dict[str, str]

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[ToolDefinition] | None,
    ) -> "ClientToolNameMap":
        original_to_alias: dict[str, str] = {}
        alias_to_original: dict[str, str] = {}
        for tool in tools or ():
            name = tool.name.strip()
            if not name or name in original_to_alias:
                continue
            alias = f"{CLIENT_TOOL_NAME_PREFIX}{name}"
            original_to_alias[name] = alias
            alias_to_original[alias] = name
        return cls(
            original_to_alias=original_to_alias,
            alias_to_original=alias_to_original,
        )

    @property
    def aliases(self) -> frozenset[str]:
        return frozenset(self.alias_to_original)

    def alias_for(self, name: str) -> str:
        return self.original_to_alias.get(name, name)

    def original_for(self, name: str) -> str:
        return self.alias_to_original.get(name, name)

    def alias_tools(
        self,
        tools: Sequence[ToolDefinition] | None,
    ) -> list[ToolDefinition] | None:
        if not tools:
            return None
        return [replace(tool, name=self.alias_for(tool.name)) for tool in tools]

    def alias_tool_choice(self, tool_choice: ToolChoice | None) -> ToolChoice | None:
        if tool_choice is None or tool_choice.mode != "function" or not tool_choice.name:
            return tool_choice
        return replace(tool_choice, name=self.alias_for(tool_choice.name))


__all__ = ["CLIENT_TOOL_NAME_PREFIX", "ClientToolNameMap"]
