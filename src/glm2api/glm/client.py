from __future__ import annotations

import codecs
import gzip
import http.client
import json
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from io import BufferedReader, BytesIO
from logging import Logger
from typing import Callable, Iterator

from ..config import AppConfig, DEFAULT_CHAT_MODEL_NAME
from ..core.models import (
    TextGenerationRequest,
    TextGenerationResponse,
    TextStreamEvent,
    ToolDefinition,
)
from ..core.usage import TokenUsage, estimate_conservative_prompt_tokens
from ..infrastructure.logging import debug_dump
from .auth import GLMAccessTokenManager, build_sign
from .chat import get_model_multimodal_capability, resolve_chat_mode, resolve_networking, resolve_upstream_model
from .errors import QueueTimeoutError, UpstreamAPIError
from .events import GLMUpstreamEventAccumulator, effective_event_status, is_nonzero_status
from .tools.dsml import BLOCKED_NATIVE_TOOL_NAMES, SERVER_SIDE_TOOL_NAMES
from .translator import convert_messages_to_glm_prompt, extract_recent_user_url, messages_to_glm_payload


_EVENT_ERROR_STATUSES = frozenset({"error", "failed", "failure", "cancelled", "canceled"})


@dataclass(slots=True)
class QueueLease:
    ticket: int
    release_callback: Callable[[int], None]
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.release_callback(self.ticket)


class ConcurrentRequestQueue:
    def __init__(self, logger: Logger, wait_timeout: int, max_concurrency: int) -> None:
        self.logger = logger
        self.wait_timeout = wait_timeout
        self.max_concurrency = max(1, max_concurrency)
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._released_tickets: set[int] = set()

    def acquire(self, request_name: str) -> QueueLease:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            queue_ahead = max(0, ticket - (self._serving_ticket + self.max_concurrency) + 1)
            start = time.monotonic()

            if queue_ahead > 0:
                self.logger.info("请求进入 GLM 队列 ticket=%s ahead=%s request=%s", ticket, queue_ahead, request_name)

            while ticket >= self._serving_ticket + self.max_concurrency:
                remaining = self.wait_timeout - (time.monotonic() - start)
                if remaining <= 0:
                    # A timed-out waiter still owns a ticket in the FIFO.  Mark
                    # it released so a later request cannot be blocked forever
                    # by a ticket that will never acquire a lease.
                    self._skip_ticket(ticket)
                    raise QueueTimeoutError(
                        f"GLM 队列等待超时，前方仍有 {ticket - (self._serving_ticket + self.max_concurrency) + 1} 个请求，请稍后重试。"
                    )
                self._condition.wait(timeout=remaining)

            active_slots = ticket - self._serving_ticket + 1
            self.logger.info(
                "请求获得 GLM 执行槽位 ticket=%s active=%s/%s request=%s",
                ticket,
                active_slots,
                self.max_concurrency,
                request_name,
            )
            return QueueLease(ticket=ticket, release_callback=self._release)

    def _release(self, ticket: int) -> None:
        with self._condition:
            self._skip_ticket(ticket)
            self.logger.info("请求离开 GLM 执行槽位 ticket=%s", ticket)
            self._condition.notify_all()

    def _skip_ticket(self, ticket: int) -> None:
        """Advance the FIFO over a released or cancelled ticket."""
        self._released_tickets.add(ticket)
        while self._serving_ticket in self._released_tickets:
            self._released_tickets.remove(self._serving_ticket)
            self._serving_ticket += 1
        self._condition.notify_all()


class GLMWebClient:
    def __init__(self, config: AppConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self.auth = GLMAccessTokenManager(config=config, logger=logger)
        self.request_queue = ConcurrentRequestQueue(
            logger=logger,
            wait_timeout=config.glm_queue_wait_timeout,
            max_concurrency=config.glm_max_concurrency,
        )
        from ..media.images import ImageService
        from ..media.videos import VideoService

        self.images = ImageService(self)
        self.videos = VideoService(self)
        from .files import FileService

        self.files = FileService(self)

    def generate_images(self, request):
        return self.images.generate(request)

    def download_image_as_base64(self, image_url: str) -> str:
        return self.images.download_as_base64(image_url)

    def create_video(self, request):
        return self.videos.create(request)

    def retrieve_video(self, video_id: str) -> dict[str, object]:
        return self.videos.retrieve(video_id)

    def list_videos(self) -> dict[str, object]:
        return self.videos.list()

    def open_video_content(self, video_id: str):
        return self.videos.open_content(video_id)

    def _resolve_tools(self, request: TextGenerationRequest) -> tuple[list[ToolDefinition] | None, set[str] | None]:
        raw_tools = list(request.tools)
        blocked_tool_names = {
            name.strip()
            for name in self.config.blocked_tool_names
            if name.strip()
        } | BLOCKED_NATIVE_TOOL_NAMES
        filtered_tools = [tool for tool in raw_tools if tool.name not in blocked_tool_names]
        blocked_names = [tool.name for tool in raw_tools if tool.name in blocked_tool_names]
        if blocked_names:
            self.logger.info("已过滤不受支持的工具: %s", ", ".join(blocked_names))
        tool_names = {tool.name for tool in filtered_tools if tool.name}
        tool_choice = request.tool_choice
        if tool_choice is not None:
            if tool_choice.mode == "none":
                return None, set()
            if tool_choice.mode in {"required", "function"} and not filtered_tools:
                raise ValueError("tool_choice 要求工具，但请求没有可用工具")
            if tool_choice.mode == "function":
                if tool_choice.name not in tool_names:
                    raise ValueError(f"tool_choice 指定的工具不可用: {tool_choice.name}")
        return filtered_tools or None, tool_names or None

    def _validate_generation_parameters(self, request: TextGenerationRequest) -> None:
        unsupported: list[str] = []
        if request.temperature is not None:
            unsupported.append("temperature")
        if request.top_p is not None:
            unsupported.append("top_p")
        if request.stop not in (None, "", ()):
            unsupported.append("stop/stop_sequences")
        if request.response_format is not None:
            unsupported.append("response_format")
        if unsupported:
            raise ValueError(
                "当前 ChatGLM 网页协议不支持生成参数: "
                + ", ".join(unsupported)
                + "；请移除这些参数后重试"
            )
        if request.max_tokens is not None:
            if (
                isinstance(request.max_tokens, bool)
                or not isinstance(request.max_tokens, int)
                or request.max_tokens < 0
            ):
                raise ValueError("max_tokens 必须是非负整数")
            # Anthropic requires max_tokens in every Messages request.  The
            # observed web protocol has no equivalent field, so keep the
            # request compatible and enforce the output budget locally.
            self.logger.warning(
                "ChatGLM 网页协议未提供 max_tokens 上游字段，当前保留兼容值并由代理本地限制输出"
            )

    def _validate_model_content(
        self,
        request: TextGenerationRequest,
        model: str,
        resolved_model: str | None = None,
    ) -> None:
        capability_model = resolved_model if resolved_model is not None else model
        if get_model_multimodal_capability(capability_model) is not False:
            return

        for message_index, message in enumerate(request.messages):
            if not isinstance(message.content, tuple):
                continue
            for block_index, block in enumerate(message.content):
                if block.kind in {"text", "thinking", "redacted_thinking"}:
                    continue
                raise ValueError(
                    f"模型 {model} 仅支持文本输入："
                    f"messages[{message_index}].content[{block_index}] 类型为 {block.kind or 'unknown'}"
                )

    def chat_completion(
        self,
        request: TextGenerationRequest,
    ) -> tuple[TextGenerationResponse, str | None]:
        _, allowed_tool_names = self._resolve_tools(request)
        lease = self.request_queue.acquire(f"chat:{request.model}")
        try:
            response, assistant_id, usage = self._open_chat_stream(
                request,
                preferred_account_index=self.get_preferred_account_index(lease.ticket),
            )
        except Exception:
            lease.release()
            raise
        accumulator = GLMUpstreamEventAccumulator(
            model=request.model,
            allowed_tool_names=allowed_tool_names,
            tool_choice=request.tool_choice,
            fallback_tool_url=extract_recent_user_url(
                messages_to_glm_payload(request.messages)
            ),
            debug_enabled=self.config.debug_dump_all,
            logger=self.logger,
            usage=usage,
            max_output_tokens=request.max_tokens,
        )
        try:
            for event in self.iter_sse_events(response):
                if not event:
                    continue
                self.raise_for_event_error(event, stream=False)
                _, status = accumulator.consume_event(event)
                if status in {"finish", "intervene"}:
                    return accumulator.build_response(), accumulator.conversation_id
        finally:
            response.close() # type: ignore
            self.delete_conversation(accumulator.conversation_id, assistant_id=assistant_id)
            lease.release()
        return accumulator.build_response(), accumulator.conversation_id

    def stream_chat_completion(
        self,
        request: TextGenerationRequest,
    ) -> Iterator[TextStreamEvent]:
        """Yield protocol-neutral text stream events."""
        request.stream = True
        _, allowed_tool_names = self._resolve_tools(request)
        lease = self.request_queue.acquire(f"stream:{request.model}")
        try:
            response, assistant_id, usage = self._open_chat_stream(
                request,
                preferred_account_index=self.get_preferred_account_index(lease.ticket),
            )
        except Exception:
            lease.release()
            raise

        accumulator = GLMUpstreamEventAccumulator(
            model=request.model,
            allowed_tool_names=allowed_tool_names,
            tool_choice=request.tool_choice,
            fallback_tool_url=extract_recent_user_url(
                messages_to_glm_payload(request.messages)
            ),
            debug_enabled=self.config.debug_dump_all,
            logger=self.logger,
            usage=usage,
            max_output_tokens=request.max_tokens,
        )
        request.usage = usage

        def generate():
            try:
                for event in self.iter_sse_events(response, require_done=True):
                    if not event:
                        continue
                    self.raise_for_event_error(event, stream=True)
                    stream_events, status = accumulator.consume_event(event)
                    yield from stream_events

                    if status in {"finish", "intervene"}:
                        yield from accumulator.finalize(
                            status=status,
                            last_error=event.get("last_error") if isinstance(event.get("last_error"), dict) else None,
                        )
                        return
                    if accumulator.output_budget_exhausted:
                        yield from accumulator.finalize(status="length")
                        return

                yield from accumulator.finalize(status="stop")
            finally:
                response.close() # type: ignore
                self.delete_conversation(accumulator.conversation_id, assistant_id=assistant_id)
                lease.release()

        return generate()

    def raise_for_event_error(self, event: dict[str, object], stream: bool) -> None:
        raw_status = effective_event_status(event)
        status = str(raw_status).strip().lower() if raw_status is not None else ""
        last_error = event.get("last_error")
        event_error = self._extract_event_error(event)
        numeric_failure = is_nonzero_status(raw_status)
        if (
            status not in _EVENT_ERROR_STATUSES
            and not numeric_failure
            and not event_error
            and not isinstance(last_error, dict)
        ):
            return

        error_payload: dict[str, object] = {}
        if isinstance(last_error, dict):
            error_payload.update(last_error)
        if isinstance(event_error, dict):
            error_payload.update(event_error)
        if (
            not error_payload
            and status not in _EVENT_ERROR_STATUSES
            and not numeric_failure
        ):
            return

        error_code = error_payload.get("error_code", error_payload.get("code"))
        error_message = str(
            error_payload.get("err_msg")
            or error_payload.get("message")
            or ("GLM stream request error" if stream else "GLM request error")
        ).strip()
        detail = f"code={error_code} " if error_code is not None else ""
        raise UpstreamAPIError(
            status_code=502,
            message=f"GLM 上游返回错误 | {detail}{error_message}".strip(),
            payload=error_payload or event,
        )

    def _extract_event_error(self, event: dict[str, object]) -> dict[str, object] | None:
        top_level_error = event.get("error")
        if isinstance(top_level_error, dict) and top_level_error:
            return top_level_error
        parts = event.get("parts")
        if not isinstance(parts, list):
            return None
        for part in parts:
            if not isinstance(part, dict):
                continue
            error = part.get("error")
            if isinstance(error, dict) and error:
                return error
            part_status = str(part.get("status", "")).strip().lower()
            if part_status == "error":
                return {"message": "GLM part status error"}
        return None

    def build_signed_request(
        self,
        url: str,
        *,
        method: str,
        access_token: str,
        data: bytes | None = None,
        app_fr: str | None = None,
        referer: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> urllib.request.Request:
        """Build a signed request shared by the GLM text and media flows."""
        timestamp, nonce, sign = build_sign()
        browser_headers = (
            self.auth.get_browser_headers()
            if app_fr is None
            else self.auth.get_browser_headers(app_fr=app_fr)
        )
        headers = {
            **browser_headers,
            "Authorization": f"Bearer {access_token}",
            "X-Device-Id": uuid.uuid4().hex,
            "X-Nonce": nonce,
            "X-Request-Id": uuid.uuid4().hex,
            "X-Sign": sign,
            "X-Timestamp": timestamp,
        }
        if referer:
            headers["Referer"] = referer
        if extra_headers:
            headers.update(extra_headers)
        return urllib.request.Request(url, method=method, data=data, headers=headers)

    def open_upstream_request(self, request: urllib.request.Request, *, opener=None):
        """Open a signed GLM request and normalize HTTP errors."""
        try:
            if opener is None:
                return urllib.request.urlopen(request, timeout=self.config.request_timeout)
            return opener.open(request, timeout=self.config.request_timeout)
        except urllib.error.HTTPError as exc:
            error_payload = self.read_error_payload(exc)
            message = self.build_error_message(exc.code, error_payload)
            raise UpstreamAPIError(status_code=exc.code, message=message, payload=error_payload) from exc

    def delete_conversation(self, conversation_id: str, assistant_id: str | None = None) -> None:
        if not self.config.glm_delete_conversation:
            return
        if not conversation_id:
            self.logger.warning("跳过删除 GLM 会话：未获取到 conversation_id assistant_id=%s", assistant_id or self.config.glm_assistant_id)
            return

        actual_assistant_id = assistant_id or self.config.glm_assistant_id
        body = json.dumps(
            {
                "assistant_id": actual_assistant_id,
                "conversation_id": conversation_id,
            }
        ).encode("utf-8")
        try:
            def send_request(account_index: int, access_token: str):
                request = self.build_signed_request(
                    self.config.delete_conversation_url,
                    method="POST",
                    access_token=access_token,
                    data=body,
                    referer="https://chatglm.cn/main/alltoolsdetail",
                )
                return urllib.request.urlopen(request, timeout=self.config.request_timeout)

            with self.call_with_account_failover("delete_conversation", send_request) as response: # type: ignore
                payload = self.auth.read_json_response(response)
            status = payload.get("status", payload.get("code"))
            if status not in {0, None}:
                self.logger.warning(
                    "GLM 会话删除返回非成功状态 conversation_id=%s assistant_id=%s payload=%s",
                    conversation_id,
                    actual_assistant_id,
                    payload,
                )
                return
            self.logger.info(
                "已删除 GLM 会话 conversation_id=%s assistant_id=%s",
                conversation_id,
                actual_assistant_id,
            )
        except Exception as exc:
            self.logger.warning(
                "删除 GLM 会话失败 conversation_id=%s assistant_id=%s error=%s",
                conversation_id,
                actual_assistant_id,
                exc,
            )

    def _open_chat_stream(
        self,
        request: TextGenerationRequest,
        preferred_account_index: int | None = None,
    ):
        requested_model = request.model or DEFAULT_CHAT_MODEL_NAME
        upstream_model, assistant_id = resolve_upstream_model(requested_model, self.config)
        self._validate_model_content(request, requested_model, upstream_model)
        self._validate_generation_parameters(request)
        filtered_tools, _ = self._resolve_tools(request)
        converted_messages = convert_messages_to_glm_prompt(
            messages=request.messages,
            tools=filtered_tools,
            blocked_tool_names={name.strip() for name in self.config.blocked_tool_names if name.strip()},
            tool_choice=request.tool_choice,
            server_side_tool_names=SERVER_SIDE_TOOL_NAMES,
        )
        debug_dump(self.logger, self.config.debug_dump_all, "内部文本请求", request)
        debug_dump(self.logger, self.config.debug_dump_all, "转换后的 GLM messages", converted_messages)
        chat_messages = messages_to_glm_payload(request.messages)
        refs = self.files.upload_referenced_files(chat_messages)
        if refs:
            converted_messages[0]["content"] = refs + list(converted_messages[0]["content"]) # type: ignore
            debug_dump(self.logger, self.config.debug_dump_all, "附加上传引用后的 GLM messages", converted_messages)
        usage = TokenUsage.estimated(
            input_tokens=max(
                estimate_conservative_prompt_tokens(converted_messages),
                estimate_conservative_prompt_tokens(chat_messages),
            )
        )

        chat_mode = resolve_chat_mode(
            model=requested_model,
            reasoning_effort=request.reasoning_effort,
            deep_research=request.deep_research,
        )
        is_networking = resolve_networking(
            model=requested_model,
            web_search=request.web_search,
        )

        request_body = json.dumps(
            {
                "assistant_id": assistant_id,
                "conversation_id": "",
                "project_id": "",
                "chat_type": "user_chat",
                "messages": converted_messages,
                "meta_data": {
                    "channel": "",
                    "chat_mode": chat_mode,
                    "draft_id": "",
                    "input_question_type": "xxxx",
                    "is_networking": is_networking,
                    "is_test": False,
                    "platform": "pc",
                    "quote_log_id": "",
                    "selected_model": upstream_model,
                    "cogview": {"rm_label_watermark": False},
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        self.logger.info(
            "转发请求 model=%s upstream=%s stream=%s",
            requested_model,
            upstream_model,
            request.stream,
        )
        debug_dump(self.logger, self.config.debug_dump_all, "转发到 GLM 的 chat 原始请求体", request_body)

        def send_request(account_index: int, access_token: str):
            for attempt in range(self.config.glm_busy_max_retries + 1):
                try:
                    request = self.build_signed_request(
                        self.config.chat_stream_url,
                        data=request_body,
                        method="POST",
                        access_token=access_token,
                    )
                    debug_dump(
                        self.logger,
                        self.config.debug_dump_all,
                        f"转发到 GLM 的 chat 请求头 account={account_index} attempt={attempt + 1}",
                        dict(request.header_items()),
                    )
                    return self._prepare_chat_response(
                        urllib.request.urlopen(request, timeout=self.config.request_timeout)
                    )
                except urllib.error.HTTPError as exc:
                    error_payload = self.read_error_payload(exc)
                    if self._should_retry_busy_error(exc.code, error_payload) and attempt < self.config.glm_busy_max_retries:
                        wait_seconds = self.config.glm_busy_retry_interval
                        self.logger.warning(
                            "GLM 正在处理其他对话，等待重试 attempt=%s/%s wait=%.1fs account=%s",
                            attempt + 1,
                            self.config.glm_busy_max_retries,
                            wait_seconds,
                            account_index,
                        )
                        time.sleep(wait_seconds)
                        continue

                    message = self.build_error_message(exc.code, error_payload)
                    raise UpstreamAPIError(status_code=exc.code, message=message, payload=error_payload) from exc

            raise UpstreamAPIError(status_code=429, message="GLM 长时间忙碌，请稍后重试。")

        response = self.call_with_account_failover(
            f"chat:{requested_model}",
            send_request,
            preferred_account_index=preferred_account_index,
        )
        request.usage = usage
        return response, assistant_id, usage

    def _prepare_chat_response(self, response):
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" in content_type:
            try:
                payload = self.auth.read_json_response(response)
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            debug_dump(self.logger, self.config.debug_dump_all, "GLM 非流式原始 JSON 响应", payload)
            status = payload.get("status")
            message = str(payload.get("message", "")).strip()
            if status not in (0, None) or message:
                raise UpstreamAPIError(
                    status_code=502,
                    message=self.build_error_message(200, payload),
                    payload=payload,
                )

            response_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            return BufferedReader(BytesIO(response_body))

        return self.wrap_stream_response(response)

    def iter_sse_events(self, response, *, require_done: bool = False):
        """Yield parsed SSE data and optionally reject a premature EOF."""
        pending = ""
        decoder = codecs.getincrementaldecoder("utf-8")("ignore")
        saw_done = False

        def emit_block(block: str):
            lines = [line for line in block.split("\n") if line.startswith("data:")]
            if not lines:
                return None
            payload = "\n".join(line[5:].strip() for line in lines)
            debug_dump(self.logger, self.config.debug_dump_all, "GLM 原始 SSE block", block)
            if payload == "[DONE]":
                return "[DONE]"
            try:
                parsed = json.loads(payload)
                debug_dump(self.logger, self.config.debug_dump_all, "GLM 解析后的 SSE payload", parsed)
                return parsed
            except json.JSONDecodeError:
                self.logger.debug("忽略无法解析的 SSE 片段: %s", payload)
                return None

        while True:
            stop_after_chunk = False
            try:
                raw_chunk = response.read(4096)
            except http.client.IncompleteRead as exc:
                raw_chunk = exc.partial or b""
                stop_after_chunk = True
                self.logger.warning("上游 SSE 连接提前断开，按已接收内容收尾 bytes=%s", len(raw_chunk))
            if not raw_chunk:
                break

            pending += decoder.decode(raw_chunk, False).replace("\r\n", "\n")

            while "\n\n" in pending:
                block, pending = pending.split("\n\n", 1)
                event = emit_block(block.strip())
                if event == "[DONE]":
                    saw_done = True
                    return
                if event is not None:
                    yield event

            if stop_after_chunk:
                break

        remaining = decoder.decode(b"", True)
        if remaining:
            pending += remaining

        if pending.strip():
            event = emit_block(pending.strip())
            if event == "[DONE]":
                saw_done = True
            elif event is not None:
                yield event

        if require_done and not saw_done:
            raise UpstreamAPIError(status_code=502, message="GLM 上游 SSE 连接提前结束。")

    def wrap_stream_response(self, response):
        content_encoding = response.headers.get("Content-Encoding", "").lower()
        if content_encoding == "gzip":
            return BufferedReader(gzip.GzipFile(fileobj=response))
        return response

    def read_error_payload(self, error: urllib.error.HTTPError) -> dict[str, object]:
        try:
            raw_body = error.read()
            content_encoding = error.headers.get("Content-Encoding", "").lower()

            if content_encoding == "gzip":
                raw_body = gzip.decompress(raw_body)

            text = raw_body.decode("utf-8", errors="ignore")
        except Exception as exc:
            return {"message": f"读取上游错误响应失败: {exc}"}
        finally:
            close = getattr(error, "close", None)
            if callable(close):
                close()
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {"message": text}

    def _should_retry_busy_error(self, status_code: int, payload: dict[str, object]) -> bool:
        if status_code != 429:
            return False
        message = str(payload.get("message", ""))
        inner_status = payload.get("status")
        return inner_status == 10061 or "请等待其他对话生成完毕" in message

    def build_error_message(self, status_code: int, payload: dict[str, object]) -> str:
        message = str(payload.get("message", "")).strip()
        inner_status = payload.get("status")
        rid = payload.get("rid")
        parts = [f"GLM 请求失败 HTTP {status_code}"]
        if inner_status is not None:
            parts.append(f"status={inner_status}")
        if message:
            parts.append(message)
        if rid:
            parts.append(f"rid={rid}")
        return " | ".join(parts)

    def get_preferred_account_index(self, ticket: int) -> int | None:
        account_count = self.auth.get_account_count()
        if account_count <= 0:
            return None
        return ticket % account_count

    def call_with_account_failover(
        self,
        request_name: str,
        operation: Callable[[int, str], object],
        preferred_account_index: int | None = None,
    ):
        account_count = self.auth.get_account_count()
        if account_count <= 0:
            raise RuntimeError("没有可用的 GLM 账号或游客 token 配置")
        start_index = preferred_account_index % account_count if preferred_account_index is not None else self.auth.get_current_account_index()
        last_exc: Exception | None = None

        for offset in range(account_count):
            account_index = (start_index + offset) % account_count
            guest_retry_limit = self.config.glm_guest_max_retries if self.auth.is_guest_account(account_index) else 0
            for attempt in range(guest_retry_limit + 1):
                try:
                    access_token = self.auth.get_access_token_for_account(account_index)
                    return operation(account_index, access_token)
                except Exception as exc:
                    last_exc = exc
                    should_switch = self.auth.should_switch_account(exc)
                    if should_switch:
                        self.auth.invalidate_account(account_index)
                    if should_switch and attempt < guest_retry_limit:
                        self.logger.warning(
                            "游客账号请求失败，重新获取游客 ck 重试 attempt=%s/%s request=%s account=%s error=%s",
                            attempt + 1,
                            guest_retry_limit,
                            request_name,
                            account_index,
                            exc,
                        )
                        continue
                    if not should_switch or account_count == 1:
                        raise
                    self.auth.advance_account(account_index, f"{request_name}: {exc}")
                    break

        self.auth.reset_account_cycle()
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"账号轮换失败：{request_name}")

__all__ = [
    "ConcurrentRequestQueue",
    "GLMWebClient",
    "QueueLease",
    "QueueTimeoutError",
    "UpstreamAPIError",
]
