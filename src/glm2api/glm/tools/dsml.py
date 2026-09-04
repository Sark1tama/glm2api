from __future__ import annotations

import json
import re

from ...utils.json import safe_json_dumps


CANONICAL_TOOL_CALL_EXAMPLE = "\n".join(
    [
        "<|DSML|tool_calls>",
        '  <|DSML|invoke name="TOOL_NAME">',
        '    <|DSML|parameter name="actual_parameter_name"><![CDATA[value]]></|DSML|parameter>',
        "  </|DSML|invoke>",
        "</|DSML|tool_calls>",
    ]
)


def normalize_tool_name(name: object) -> str:
    return str(name).strip()


def filter_tools(tools: list[dict[str, object]] | None, blocked_tool_names: set[str]) -> list[dict[str, object]] | None:
    if not tools:
        return None

    filtered_tools: list[dict[str, object]] = []
    for tool in tools:
        fn = tool.get("function", {})
        tool_name = normalize_tool_name(fn.get("name", ""))  # type: ignore[union-attr]
        if not tool_name or tool_name in blocked_tool_names:
            continue
        filtered_tools.append(tool)

    return filtered_tools or None


def _xml_escape_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _xml_wrap_scalar(value: object) -> str:
    if isinstance(value, str):
        return f"<![CDATA[{value.replace(']]>', ']]]]><![CDATA[>')}]]>"
    return safe_json_dumps(value)


def _safe_parameter_name(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]", "_", str(value).strip()) or "value"


def _dsml_parameters_from_object(payload: object) -> str:
    if isinstance(payload, dict):
        parts: list[str] = []
        for key, value in payload.items():
            name = _xml_escape_text(_safe_parameter_name(key))
            parts.append(f'<|DSML|parameter name="{name}">{_dsml_parameters_from_object(value)}</|DSML|parameter>')
        return "".join(parts)
    if isinstance(payload, list):
        return "".join(f"<item>{_dsml_parameters_from_object(item)}</item>" for item in payload)
    return _xml_wrap_scalar(payload)


def serialize_tool_call_block(name: str, arguments: object) -> str:
    parsed_arguments = arguments
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_arguments = {"raw": arguments}
    if not isinstance(parsed_arguments, dict):
        parsed_arguments = {"value": parsed_arguments}
    return (
        "<|DSML|tool_calls>\n"
        f'  <|DSML|invoke name="{_xml_escape_text(name)}">\n'
        f"    {_dsml_parameters_from_object(parsed_arguments)}\n"
        "  </|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )


def serialize_tool_result_block(tool_call_id: object, tool_name: str, content: str) -> str:
    safe_content = content.replace("]]>", "]]]]><![CDATA[>")
    return (
        f'<|DSML|tool_result call_id="{_xml_escape_text(str(tool_call_id or "unknown"))}" '
        f'name="{_xml_escape_text(tool_name)}"><content><![CDATA[{safe_content}]]></content></|DSML|tool_result>'
    )


def build_tool_call_instructions(
    tool_names: list[str],
    tool_choice_policy: dict[str, object] | None = None,
) -> str:
    available_xml_names = ", ".join(f"`{name}`" for name in tool_names) or "`(none)`"

    policy = tool_choice_policy or {"mode": "auto", "tool_name": None}
    mode = str(policy.get("mode", "auto"))
    specific_name = str(policy.get("tool_name", "") or "")
    lines = [
        "# TOOL USE PROTOCOL",
        "There are two separate execution environments.",
        "Provider-side tools run inside ChatGLM's remote environment or access public web resources.",
        "The tool schemas below are client-side tools that run in the user's environment.",
        "Client-side tools are available through DSML even when they do not appear in the provider's native tool list.",
        "Do not claim that a listed client-side tool is unavailable merely because it is absent from the provider's native tool list.",
        "To call a client-side tool, emit its exact declared name and arguments using DSML.",
        "A tool name emitted through DSML always refers to the listed client-side tool, even when a provider-side tool has the same name.",
    ]

    if tool_names:
        lines.extend(
            [
                "",
                f"Client-side DSML tools: {available_xml_names}.",
                "Only these names may be emitted as DSML tools. Use the exact parameter fields from their schemas.",
                "A client tool call must be one executable DSML block in the final assistant text. Do not mix it with prose or another tool-call syntax.",
                "Use the DSML format below exactly.",
                CANONICAL_TOOL_CALL_EXAMPLE,
                "Parameter rules:",
                "- The root executable block must be <|DSML|tool_calls> and each call must be a <|DSML|invoke name=\"...\"> child.",
                "- Each argument must be a <|DSML|parameter name=\"...\"> child of the invoke.",
                "- Parameter names are case-sensitive and must exactly match the schema. For example, use `filePath` only when the schema says `filePath`; never change it to `filepath`, `file_path`, or `FilePath`.",
                "- Encode nested objects with nested <|DSML|parameter name=\"...\"> tags.",
                "- Use repeated <item> tags to represent arrays.",
                "- JSON literals are allowed as parameter values when the schema expects an object, array, number, boolean, or null.",
                "- Prefer <![CDATA[...]]> for arbitrary strings.",
            ]
        )

    lines.extend(
        [
            "",
            "Rules:",
            "- Do not emit undeclared names as DSML tools.",
            "- Provider-side tools may be used internally and must not be represented as client DSML calls.",
            "- Access to the user's files, directories, repositories, shell, processes, configuration, services, hardware, or localhost requires an appropriate listed client-side tool through DSML.",
            "- Never use a provider-side tool or remote sandbox as a substitute for access to the user's environment.",
            "- Provider-side web tools may be used for public internet resources.",
            "- Do not draft or hide a client tool call only in reasoning. When a client call is needed, emit its executable DSML block in the final assistant text.",
            "- Do not emit OpenAI JSON tool_calls arrays or function_call objects for client tools.",
            "- Put multiple DSML invokes inside one <|DSML|tool_calls> root when you truly need multiple calls in one turn.",
            "- After a <|DSML|tool_result ...> block, continue from that result and call another tool only when necessary.",
        ]
    )
    if mode == "none":
        lines.extend(
            [
                "Tool choice policy: none.",
                "Do not emit any executable tool markup. Answer with normal text only.",
            ]
        )
    elif mode == "required":
        lines.extend(
            [
                "Tool choice policy: required.",
                "You must call at least one listed client-side tool through DSML before giving a final answer.",
            ]
        )
    elif mode == "specific" and specific_name:
        lines.extend(
            [
                "Tool choice policy: specific function.",
                f"You must call exactly the client-side tool `{specific_name}` through DSML before giving a final answer.",
                f"Do not call any other client-side tool.",
            ]
        )
    else:
        lines.extend(
            [
                "Tool choice policy: auto.",
                "Decide whether tool execution is necessary.",
                "If the task requires access to the user's environment, use a listed client-side tool through DSML.",
                "If the user explicitly requests a listed client-side tool, treat that tool execution as necessary.",
                "Use provider-side tools only for public internet or provider-hosted resources.",
                "Answer directly only when no tool execution is needed.",
                "Do not substitute one execution environment for another.",
            ]
        )
    return "\n".join(lines)


def tools_to_prompt(
    tools: list[dict[str, object]],
    blocked_tool_names: set[str] | None = None,
    tool_choice_policy: dict[str, object] | None = None,
) -> str:
    tool_names: list[str] = []
    tool_schemas: list[str] = []
    for tool in tools:
        fn = tool.get("function", {})
        name = str(fn.get("name", "unknown"))  # type: ignore[union-attr]
        description = str(fn.get("description", "") or "")  # type: ignore[union-attr]
        parameters = fn.get("parameters", {})  # type: ignore[union-attr]
        if blocked_tool_names and name in blocked_tool_names:
            continue
        tool_names.append(name)
        tool_schemas.append(
            "\n".join(
                [
                    f"Tool: {name}",
                    f"Description: {description}",
                    f"Parameters: {safe_json_dumps(parameters) if isinstance(parameters, dict) else '{}'}",
                ]
            )
        )

    parts = [
        "# TOOL SCHEMAS",
        "Treat the following schema list as the authoritative tool contract for this request.",
        "",
        "\n\n".join(tool_schemas),
        "",
        build_tool_call_instructions(
            tool_names,
            tool_choice_policy=tool_choice_policy,
        ),
    ]
    return "\n".join(part for part in parts if part is not None).strip()
