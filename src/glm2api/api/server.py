from __future__ import annotations

import json
import queue
import threading
import traceback
import uuid
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging import Logger
from urllib.parse import urlparse

from ..config import AppConfig, DEFAULT_CHAT_MODEL_NAME, DEFAULT_MAX_REQUEST_BODY_BYTES
from ..infrastructure.logging import debug_dump
from .errors import (
    anthropic_error_type,
    build_anthropic_error_payload,
    build_error_payload,
    safe_http_status,
)
from .sse import (
    CLIENT_DISCONNECTED,
    SSEWriter,
    start_sse_response,
    write_sse_error,
)
from .adapters.anthropic.messages import (
    AnthropicMessagesStreamAccumulator,
    anthropic_messages_to_internal,
    count_anthropic_cache_control_markers,
    estimate_anthropic_input_tokens,
    internal_to_anthropic_messages_response,
    write_anthropic_sse_error,
)
from ..glm.client import GLMWebClient
from ..glm.errors import QueueTimeoutError, UpstreamAPIError
from .adapters.openai.chat_completions import (
    internal_to_openai_chat_completions_response,
    openai_chat_completions_to_internal,
    serialize_openai_chat_completions_stream_event,
)
from ..core.models import (
    TextGenerationRequest,
    TextGenerationResponse,
    TextStreamEvent,
)
from ..media.images import (
    ImageResponseConversionError,
    internal_to_openai_images_response,
    openai_images_to_internal,
)
from .adapters.openai.responses import (
    OpenAIResponsesStreamAccumulator,
    internal_to_openai_responses_response,
    openai_responses_to_internal,
    write_responses_sse_error,
)
from ..media.videos import (
    VIDEO_MAX_UPLOAD_BYTES,
    VideoNotFoundError,
    VideoNotReadyError,
    normalize_video_request,
)


RESPONSES_STREAM_HEARTBEAT_SECONDS = 5.0


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class GLM2APIServer:
    def __init__(self, config: AppConfig, glm_client: GLMWebClient, logger: Logger) -> None:
        self.config = config
        self.glm_client = glm_client
        self.logger = logger
        handler_cls = self._build_handler()
        self._server = _ReusableThreadingHTTPServer((config.host, config.port), handler_cls)

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _build_handler(self):
        config = self.config
        glm_client = self.glm_client
        logger = self.logger

        class RequestHandler(BaseHTTPRequestHandler):
            server_version = "glm2api/0.1.0"
            protocol_version = "HTTP/1.1"

            def do_OPTIONS(self) -> None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self._send_common_headers()
                self.end_headers()

            def do_GET(self) -> None:
                try:
                    self._debug_log_request_start()
                    path = self._path_without_query()
                    if path == "/health":
                        self._write_json(HTTPStatus.OK, {"status": "ok"})
                        return

                    if path == f"{config.api_prefix}/models":
                        self._write_json(
                            HTTPStatus.OK,
                            {
                                "object": "list",
                                "data": [
                                    {"id": model, "object": "model", "owned_by": "glm2api"}
                                    for model in config.exposed_models
                                ],
                            },
                        )
                        return

                    video_path_prefix = f"{config.api_prefix}/videos"
                    if path == video_path_prefix or path.startswith(f"{video_path_prefix}/"):
                        if not self._authorize():
                            logger.warning("认证失败 path=%s ip=%s", self.path, self.client_address[0])
                            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": {"message": "Unauthorized"}})
                            return

                    if path == video_path_prefix:
                        self._write_json(HTTPStatus.OK, glm_client.list_videos())
                        return

                    video_item_path_prefix = f"{video_path_prefix}/"
                    if path.startswith(video_item_path_prefix):
                        video_path = path[len(video_item_path_prefix):]
                        video_id, separator, suffix = video_path.partition("/")
                        if not video_id or (separator and suffix != "content"):
                            self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})
                            return
                        if separator:
                            self._stream_video_content(video_id)
                        else:
                            self._write_json(HTTPStatus.OK, glm_client.retrieve_video(video_id))
                        return

                    logger.debug("GET 未匹配 path=%s", self.path)
                    self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})
                except CLIENT_DISCONNECTED:
                    logger.warning("客户端在 GET 响应写回前断开 path=%s", self.path)
                except VideoNotFoundError as exc:
                    self._safe_write_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": {"message": str(exc), "type": "not_found"}},
                    )
                except VideoNotReadyError as exc:
                    self._safe_write_json(
                        HTTPStatus.CONFLICT,
                        {"error": {"message": str(exc), "type": "video_not_ready"}},
                    )
                except UpstreamAPIError as exc:
                    logger.warning("视频上游请求失败 status=%s error=%s", exc.status_code, exc)
                    self._safe_write_json(
                        safe_http_status(exc.status_code, fallback=HTTPStatus.BAD_GATEWAY),
                        {"error": {"message": str(exc), "type": "upstream_error", "details": exc.payload}},
                    )
                except Exception as exc:
                    logger.error("处理 GET 请求失败 path=%s error=%s\n%s", self.path, exc, traceback.format_exc())
                    self._safe_write_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": {"message": "服务内部错误", "type": exc.__class__.__name__}},
                    )

            def do_POST(self) -> None:
                self._request_id = f"req_{uuid.uuid4().hex}"
                self._request_path = ""
                try:
                    self._debug_log_request_start()
                    path = self._path_without_query()
                    self._request_path = path
                    if path not in {
                        f"{config.api_prefix}/chat/completions",
                        f"{config.api_prefix}/images/generations",
                        f"{config.api_prefix}/videos",
                        f"{config.api_prefix}/messages",
                        f"{config.api_prefix}/messages/count_tokens",
                        f"{config.api_prefix}/responses",
                    }:
                        logger.debug("POST 未匹配 path=%s", self.path)
                        self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})
                        return

                    if not self._authorize():
                        logger.warning("认证失败 path=%s ip=%s", self.path, self.client_address[0])
                        self._write_request_error(
                            HTTPStatus.UNAUTHORIZED,
                            "Unauthorized",
                            "unauthorized",
                            legacy_payload={"error": {"message": "Unauthorized"}},
                        )
                        return

                    transfer_encoding = self.headers.get("Transfer-Encoding", "").strip().lower()
                    if transfer_encoding and transfer_encoding != "identity":
                        self._write_request_error(
                            HTTPStatus.LENGTH_REQUIRED,
                            "当前 HTTP 入口需要 Content-Length，不支持 Transfer-Encoding。",
                            "length_required",
                            legacy_payload={
                                "error": {
                                    "message": "当前 HTTP 入口需要 Content-Length，不支持 Transfer-Encoding。",
                                    "type": "length_required",
                                }
                            },
                        )
                        return
                    content_length = self._parse_content_length()
                    if content_length < 0:
                        self._write_request_error(
                            HTTPStatus.BAD_REQUEST,
                            "Content-Length 不能为负数。",
                            "invalid_content_length",
                            legacy_payload={
                                "error": {"message": "Content-Length 不能为负数。", "type": "invalid_content_length"}
                            },
                        )
                        return
                    max_request_body_bytes = getattr(
                        config, "max_request_body_bytes", DEFAULT_MAX_REQUEST_BODY_BYTES
                    )
                    if content_length > max_request_body_bytes:
                        self._write_json(
                            HTTPStatus(413),
                            {
                                "error": {
                                    "message": "请求体超过服务端大小限制。",
                                    "type": "request_too_large",
                                }
                            },
                        )
                        return
                    if path == f"{config.api_prefix}/videos" and content_length > VIDEO_MAX_UPLOAD_BYTES + 1024 * 1024:
                        self._write_json(
                            HTTPStatus(413),
                            {"error": {"message": "视频请求体超过 101MB。", "type": "request_too_large"}},
                        )
                        return
                    raw_body = self.rfile.read(content_length) if content_length else b"{}"

                    if path == f"{config.api_prefix}/videos":
                        payload = self._parse_video_request(raw_body)
                        debug_dump(
                            logger,
                            config.debug_dump_all,
                            f"HTTP 入站视频请求 path={self.path}",
                            self._redact_video_payload(payload),
                        )
                        logger.info("收到视频请求 model=%s prompt=%s", payload.get("model"), payload.get("prompt"))
                        result = glm_client.create_video(normalize_video_request(payload))
                        self._write_json(HTTPStatus.OK, result)
                        return

                    debug_dump(logger, config.debug_dump_all, f"HTTP 入站原始请求体 path={self.path}", raw_body)
                    try:
                        payload = json.loads(raw_body.decode("utf-8"))
                    except UnicodeDecodeError:
                        self._write_request_error(
                            HTTPStatus.BAD_REQUEST,
                            "请求体必须是 UTF-8 编码。",
                            "invalid_encoding",
                            legacy_payload={
                                "error": {"message": "请求体必须是 UTF-8 编码。", "type": "invalid_encoding"}
                            },
                        )
                        return
                    except json.JSONDecodeError as exc:
                        self._write_request_error(
                            HTTPStatus.BAD_REQUEST,
                            f"请求体不是合法 JSON: {exc.msg}",
                            "invalid_json",
                            legacy_payload={
                                "error": {
                                    "message": f"请求体不是合法 JSON: {exc.msg}",
                                    "type": "invalid_json",
                                }
                            },
                        )
                        return

                    if not isinstance(payload, dict):
                        self._write_request_error(
                            HTTPStatus.BAD_REQUEST,
                            "请求体顶层必须是 JSON 对象。",
                            "invalid_payload",
                            legacy_payload={
                                "error": {"message": "请求体顶层必须是 JSON 对象。", "type": "invalid_payload"}
                            },
                        )
                        return
                    debug_dump(logger, config.debug_dump_all, f"HTTP 入站解析后 JSON path={self.path}", payload)

                    # --- Anthropic Messages API ---
                    if path == f"{config.api_prefix}/messages/count_tokens":
                        logger.info("收到 Anthropic count_tokens 请求 model=%s", payload.get("model"))
                        self._log_anthropic_cache_control_noop(payload)
                        self._handle_anthropic_count_tokens(payload)
                        return

                    if path == f"{config.api_prefix}/messages":
                        logger.info("收到 Anthropic 请求 model=%s stream=%s", payload.get("model"), payload.get("stream"))
                        self._log_anthropic_cache_control_noop(payload)
                        self._handle_anthropic_messages(payload)
                        return

                    # --- OpenAI Responses API ---
                    if path == f"{config.api_prefix}/responses":
                        logger.info("收到 Responses 请求 model=%s stream=%s", payload.get("model"), payload.get("stream"))
                        self._handle_responses(payload)
                        return

                    # --- Image generation ---
                    if path == f"{config.api_prefix}/images/generations":
                        request = openai_images_to_internal(
                            payload,
                            default_model=config.glm_image_model_name,
                        )
                        logger.info("收到绘图请求 model=%s prompt=%s", request.model, request.prompt)
                        result = glm_client.generate_images(request)
                        try:
                            response = internal_to_openai_images_response(
                                request,
                                result,
                                download_image=glm_client.download_image_as_base64,
                            )
                        except ImageResponseConversionError as exc:
                            raise UpstreamAPIError(
                                status_code=502,
                                message=f"GLM 图片结果无法转换: {exc}",
                            ) from exc
                        self._write_json(HTTPStatus.OK, response)
                        return

                    # --- Chat completions ---
                    if not isinstance(payload.get("messages"), list) or not payload.get("model"):
                        self._write_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": {"message": "请求体必须包含 model 和 messages 字段。"}},
                        )
                        return

                    if payload.get("stream"):
                        self._stream_completion(payload)
                        return

                    logger.info("收到 chat 请求 model=%s", payload.get("model"))
                    result, conversation_id = glm_client.chat_completion(
                        openai_chat_completions_to_internal(payload)
                    )
                    if not isinstance(result, TextGenerationResponse):
                        raise TypeError("GLM 客户端返回了非内部文本响应")
                    self._write_json(HTTPStatus.OK, internal_to_openai_chat_completions_response(result))
                except QueueTimeoutError as exc:
                    logger.warning("GLM 队列等待超时 error=%s", exc)
                    self._write_request_error(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        str(exc),
                        "queue_timeout",
                        legacy_payload={"error": {"message": str(exc), "type": "queue_timeout"}},
                    )
                except UpstreamAPIError as exc:
                    logger.warning("上游 GLM 返回错误 status=%s error=%s", exc.status_code, exc)
                    status = safe_http_status(exc.status_code, fallback=HTTPStatus.BAD_GATEWAY)
                    self._write_request_error(
                        status,
                        str(exc),
                        "upstream_error",
                        details=exc.payload,
                        legacy_payload={"error": {"message": str(exc), "type": "upstream_error", "details": exc.payload}},
                    )
                except ValueError as exc:
                    logger.warning("请求参数错误 path=%s error=%s", self.path, exc)
                    self._write_request_error(
                        HTTPStatus.BAD_REQUEST,
                        str(exc),
                        "invalid_request",
                        legacy_payload={"error": {"message": str(exc), "type": "invalid_request"}},
                    )
                except CLIENT_DISCONNECTED as exc:
                    logger.warning("客户端连接提前断开 path=%s error=%s", self.path, exc)
                except Exception as exc:
                    logger.error("处理请求失败 error=%s\n%s", exc, traceback.format_exc())
                    self._safe_write_request_error(
                        HTTPStatus.BAD_GATEWAY,
                        str(exc),
                        exc.__class__.__name__,
                        legacy_payload={"error": {"message": str(exc), "type": exc.__class__.__name__}},
                    )

            # ---- Anthropic Messages API ----

            def _log_anthropic_cache_control_noop(self, payload: dict[str, object]) -> None:
                markers = count_anthropic_cache_control_markers(payload)
                if markers:
                    logger.info(
                        "Anthropic cache_control 当前为 no-op，未向 GLM 执行缓存语义 markers=%s",
                        markers,
                    )

            def _handle_anthropic_count_tokens(self, payload: dict[str, object]) -> None:
                model = payload.get("model")
                if not isinstance(model, str) or not model.strip():
                    raise ValueError("Anthropic count_tokens 请求必须包含 model 字段。")
                if not isinstance(payload.get("messages"), list):
                    raise ValueError("Anthropic count_tokens 请求必须包含 messages 数组。")

                input_tokens = estimate_anthropic_input_tokens(payload)
                self._write_json(HTTPStatus.OK, {"input_tokens": input_tokens})

            def _handle_anthropic_messages(self, payload: dict[str, object]) -> None:
                request = anthropic_messages_to_internal(payload)
                model = request.model or DEFAULT_CHAT_MODEL_NAME

                if payload.get("stream"):
                    self._stream_anthropic(request, model)
                    return

                result, _ = glm_client.chat_completion(request)
                if not isinstance(result, TextGenerationResponse):
                    raise TypeError("GLM 客户端返回了非内部文本响应")
                response = internal_to_anthropic_messages_response(result, model)
                self._write_json(HTTPStatus.OK, response)

            def _stream_anthropic(self, request: TextGenerationRequest, model: str) -> None:
                request.stream = True
                stream_iter = glm_client.stream_chat_completion(request)
                usage = request.usage
                accumulator = AnthropicMessagesStreamAccumulator(model=model, usage=usage)

                writer = start_sse_response(
                    self,
                    self._send_common_headers,
                    request_id=self._current_request_id(),
                )

                stream_failed = False
                try:
                    for stream_item in stream_iter:
                        if not stream_item:
                            continue
                        if not isinstance(stream_item, TextStreamEvent):
                            raise TypeError("GLM 流式客户端返回了非内部流事件")
                        if not accumulator.started:
                            start_event = accumulator.start_message()
                            writer.write(start_event)
                        events = accumulator.feed_event(stream_item)
                        for event in events:
                            writer.write(event)
                except CLIENT_DISCONNECTED as exc:
                    logger.warning("客户端在 Anthropic 流式响应过程中断开 model=%s error=%s", model, exc)
                    return
                except UpstreamAPIError as exc:
                    stream_failed = True
                    logger.warning("Anthropic 流式上游失败 model=%s status=%s error=%s", model, exc.status_code, exc)
                    status = safe_http_status(exc.status_code, fallback=HTTPStatus.BAD_GATEWAY)
                    self._write_anthropic_sse_error(str(exc), anthropic_error_type(status, "upstream_error"))
                except Exception as exc:
                    stream_failed = True
                    logger.error("Anthropic 流式请求失败 model=%s error=%s\n%s", model, exc, traceback.format_exc())
                    self._write_anthropic_sse_error(str(exc), "api_error")
                finally:
                    close = getattr(stream_iter, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            logger.debug("关闭 Anthropic 上游流失败", exc_info=True)

                if not stream_failed and not accumulator.started:
                    stream_failed = True
                    self._write_anthropic_sse_error("GLM 上游未返回任何流式数据。", "api_error")

                # Only a clean stream may emit message_delta/message_stop.
                if not stream_failed and accumulator.started:
                    try:
                        for event in accumulator.finish():
                            writer.write(event)
                    except CLIENT_DISCONNECTED:
                        pass

                if stream_failed:
                    logger.warning("Anthropic 流式请求失败 model=%s", model)
                else:
                    logger.info("Anthropic 流式请求完成 model=%s", model)

            # ---- OpenAI Responses API ----

            def _handle_responses(self, payload: dict[str, object]) -> None:
                request = openai_responses_to_internal(payload)
                model = request.model or DEFAULT_CHAT_MODEL_NAME

                if payload.get("stream"):
                    self._stream_responses(request, model)
                    return

                result, _ = glm_client.chat_completion(request)
                if not isinstance(result, TextGenerationResponse):
                    raise TypeError("GLM 客户端返回了非内部文本响应")
                response = internal_to_openai_responses_response(
                    result,
                    model,
                    max_output_tokens=request.max_tokens,
                )
                self._write_json(HTTPStatus.OK, response)

            def _stream_responses(self, request: TextGenerationRequest, model: str) -> None:
                request.stream = True
                stream_iter = glm_client.stream_chat_completion(request)
                usage = request.usage
                accumulator = OpenAIResponsesStreamAccumulator(
                    model=model,
                    usage=usage,
                    max_output_tokens=request.max_tokens,
                )

                writer = start_sse_response(self, self._send_common_headers)

                chunk_queue: queue.Queue[object] = queue.Queue(maxsize=32)
                sentinel = object()
                stop_event = threading.Event()
                reader_done = threading.Event()

                def enqueue(item: object) -> bool:
                    while not stop_event.is_set():
                        try:
                            chunk_queue.put(item, timeout=0.1)
                            return True
                        except queue.Full:
                            continue
                    return False

                def read_upstream() -> None:
                    try:
                        for upstream_chunk in stream_iter:
                            if not enqueue(upstream_chunk):
                                return
                    except BaseException as exc:
                        enqueue(exc)
                    finally:
                        enqueue(sentinel)
                        close = getattr(stream_iter, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                logger.debug("关闭 Responses 上游流失败", exc_info=True)
                        reader_done.set()

                threading.Thread(target=read_upstream, daemon=True).start()

                stream_failed = False
                try:
                    while True:
                        try:
                            queued = chunk_queue.get(timeout=RESPONSES_STREAM_HEARTBEAT_SECONDS)
                        except queue.Empty:
                            writer.keepalive()
                            continue

                        if queued is sentinel:
                            break
                        if isinstance(queued, BaseException):
                            raise queued
                        stream_item = queued
                        if not stream_item:
                            continue
                        if not isinstance(stream_item, TextStreamEvent):
                            raise TypeError("GLM 流式客户端返回了非内部流事件")
                        if not accumulator.started:
                            start_events = accumulator.start_response()
                            for event in start_events:
                                writer.write(event)
                        events = accumulator.feed_event(stream_item)
                        for event in events:
                            writer.write(event)
                except CLIENT_DISCONNECTED as exc:
                    logger.warning("客户端在 Responses 流式响应过程中断开 model=%s error=%s", model, exc)
                    return
                except UpstreamAPIError as exc:
                    stream_failed = True
                    logger.warning("Responses 流式上游失败 model=%s status=%s error=%s", model, exc.status_code, exc)
                    self._write_responses_sse_error(accumulator, str(exc), "upstream_error")
                except Exception as exc:
                    stream_failed = True
                    logger.error("Responses 流式请求失败 model=%s error=%s\n%s", model, exc, traceback.format_exc())
                    self._write_responses_sse_error(accumulator, str(exc), "api_error")
                finally:
                    stop_event.set()
                    if not reader_done.wait(timeout=1.0):
                        logger.warning("Responses 上游流未在客户端断开后及时结束 model=%s", model)

                if not stream_failed and not accumulator.started:
                    stream_failed = True
                    self._write_responses_sse_error(
                        accumulator,
                        "GLM 上游未返回任何流式数据。",
                        "upstream_error",
                    )

                # Only a clean stream may emit response.completed.
                if not stream_failed and accumulator.started:
                    try:
                        for event in accumulator.finish():
                            writer.write(event)
                    except CLIENT_DISCONNECTED:
                        pass

                if stream_failed:
                    logger.warning("Responses 流式请求失败 model=%s", model)
                else:
                    logger.info("Responses 流式请求完成 model=%s", model)

            # ---- Chat completions (original) ----

            def _stream_completion(self, payload: dict[str, object]) -> None:
                request = openai_chat_completions_to_internal(payload)
                model = request.model or "unknown"
                logger.info("开始流式响应 model=%s", model)
                stream_iter = glm_client.stream_chat_completion(request)
                writer = start_sse_response(self, self._send_common_headers)

                sent_done = False
                stream_failed = False
                try:
                    for stream_item in stream_iter:
                        if not stream_item:
                            continue
                        if not isinstance(stream_item, TextStreamEvent):
                            raise TypeError("GLM 流式客户端返回了非内部流事件")
                        debug_dump(logger, config.debug_dump_all, f"HTTP 出站流式分片 model={model}", stream_item)
                        chunks = serialize_openai_chat_completions_stream_event(stream_item)
                        if stream_item.kind == "done":
                            sent_done = True
                        for chunk in chunks:
                            writer.write(chunk)
                except UpstreamAPIError as exc:
                    stream_failed = True
                    logger.warning("流式请求中途收到上游错误 status=%s error=%s", exc.status_code, exc)
                    self._write_sse_error(str(exc), "upstream_error")
                except CLIENT_DISCONNECTED as exc:
                    logger.warning("客户端在流式响应过程中断开 model=%s error=%s", model, exc)
                    return
                except Exception as exc:
                    stream_failed = True
                    logger.error("流式请求失败 model=%s error=%s\n%s", model, exc, traceback.format_exc())
                    self._write_sse_error(str(exc), exc.__class__.__name__)
                else:
                    if not sent_done:
                        stream_failed = True
                        self._write_sse_error("GLM 上游未返回流式结束标记。", "upstream_error")
                finally:
                    close = getattr(stream_iter, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            logger.debug("关闭 Chat Completions 上游流失败", exc_info=True)
                    if not sent_done:
                        try:
                            writer.done()
                        except CLIENT_DISCONNECTED:
                            pass
                if stream_failed:
                    logger.warning("流式请求失败 model=%s", model)
                else:
                    logger.info("流式请求完成 model=%s", model)

            # ---- Auth ----

            def _authorize(self) -> bool:
                if not config.server_api_keys:
                    return True
                # Support both Bearer token and x-api-key header (Anthropic style)
                authorization = self.headers.get("Authorization", "")
                if authorization.startswith("Bearer "):
                    token = authorization[7:].strip()
                    if token in config.server_api_keys:
                        return True
                x_api_key = self.headers.get("x-api-key", "")
                if x_api_key and x_api_key.strip() in config.server_api_keys:
                    return True
                return False

            # ---- Helpers ----

            def _parse_video_request(self, raw_body: bytes) -> dict[str, object]:
                content_type = self.headers.get("Content-Type", "application/json").strip()
                if content_type.lower().startswith("application/json"):
                    try:
                        payload = json.loads(raw_body.decode("utf-8"))
                    except UnicodeDecodeError as exc:
                        raise ValueError("视频请求体必须是 UTF-8 编码。") from exc
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"视频请求体不是合法 JSON: {exc.msg}") from exc
                    if not isinstance(payload, dict):
                        raise ValueError("视频请求体顶层必须是 JSON 对象。")
                    return payload

                if not content_type.lower().startswith("multipart/form-data"):
                    raise ValueError("视频接口只支持 application/json 或 multipart/form-data。")
                try:
                    envelope = (
                        b"MIME-Version: 1.0\r\nContent-Type: "
                        + content_type.encode("ascii")
                        + b"\r\n\r\n"
                        + raw_body
                    )
                    message = BytesParser(policy=email_policy).parsebytes(envelope)
                except (UnicodeEncodeError, ValueError) as exc:
                    raise ValueError("multipart 视频请求的 Content-Type 或 boundary 无效。") from exc
                if not message.is_multipart():
                    raise ValueError("multipart 视频请求格式无效。")

                payload: dict[str, object] = {}
                accepted_fields = {
                    "prompt",
                    "model",
                    "seconds",
                    "size",
                    "quality",
                    "resolution",
                    "fps",
                    "generate_audio",
                    "remove_watermark",
                    "input_reference",
                }
                for part in message.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if not isinstance(name, str) or not name:
                        continue
                    filename = part.get_filename()
                    value = part.get_payload(decode=True) or b""
                    if filename is not None or name in {"file", "input_reference"} and part.get_content_type().startswith("image/"):
                        if len(value) > VIDEO_MAX_UPLOAD_BYTES:
                            raise ValueError("input_reference 文件超过 100MB。")
                        payload["_input_reference_file"] = {
                            "filename": filename or "reference-image",
                            "mime_type": part.get_content_type(),
                            "content": value,
                        }
                        continue
                    if name in accepted_fields:
                        try:
                            payload[name] = value.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            raise ValueError(f"multipart 字段 {name} 不是 UTF-8 编码。") from exc
                return payload

            def _redact_video_payload(self, payload: dict[str, object]) -> dict[str, object]:
                redacted: dict[str, object] = {}
                for key, value in payload.items():
                    if key == "_input_reference_file" and isinstance(value, dict):
                        redacted[key] = {
                            "filename": value.get("filename"),
                            "mime_type": value.get("mime_type"),
                            "bytes": len(value.get("content", b"")) if isinstance(value.get("content"), bytes) else 0,
                        }
                    else:
                        redacted[key] = value
                return redacted

            def _stream_video_content(self, video_id: str) -> None:
                response = glm_client.open_video_content(video_id)
                try:
                    content_type = response.headers.get("Content-Type", "video/mp4") or "video/mp4"
                    content_length = response.headers.get("Content-Length")
                    self.send_response(HTTPStatus.OK)
                    self._send_common_headers()
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Disposition", "inline; filename=\"video.mp4\"")
                    self.send_header("Connection", "close")
                    if content_length:
                        self.send_header("Content-Length", content_length)
                    self.end_headers()
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                finally:
                    response.close()

            def _is_anthropic_messages_request(self) -> bool:
                path = getattr(self, "_request_path", None)
                if path is None:
                    try:
                        path = self._path_without_query()
                    except Exception:
                        return False
                return path in {
                    f"{config.api_prefix}/messages",
                    f"{config.api_prefix}/messages/count_tokens",
                }

            def _current_request_id(self) -> str:
                request_id = getattr(self, "_request_id", None)
                if not isinstance(request_id, str) or not request_id:
                    request_id = f"req_{uuid.uuid4().hex}"
                    self._request_id = request_id
                return request_id

            def _write_anthropic_error(self, status: HTTPStatus, error_type: str, message: str) -> None:
                request_id = self._current_request_id()
                body = json.dumps(
                    build_anthropic_error_payload(error_type, message, request_id),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                debug_dump(logger, config.debug_dump_all, f"HTTP 出站 Anthropic 错误响应 status={int(status)} path={self.path}", body)
                self.send_response(status)
                self._send_common_headers()
                self.send_header("request-id", request_id)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_request_error(
                self,
                status: HTTPStatus,
                message: str,
                error_type: str,
                *,
                details: object | None = None,
                legacy_payload: dict[str, object] | None = None,
            ) -> None:
                if self._is_anthropic_messages_request():
                    self._write_anthropic_error(status, anthropic_error_type(status, error_type), message)
                    return
                if legacy_payload is not None:
                    self._write_json(status, legacy_payload)
                    return

                self._write_json(status, build_error_payload(message, error_type, details))

            def _safe_write_request_error(
                self,
                status: HTTPStatus,
                message: str,
                error_type: str,
                *,
                details: object | None = None,
                legacy_payload: dict[str, object] | None = None,
            ) -> None:
                try:
                    self._write_request_error(
                        status,
                        message,
                        error_type,
                        details=details,
                        legacy_payload=legacy_payload,
                    )
                except CLIENT_DISCONNECTED:
                    logger.warning("客户端在错误响应写回前断开 path=%s", self.path)

            def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                debug_dump(logger, config.debug_dump_all, f"HTTP 出站 JSON 响应 status={int(status)} path={self.path}", body)
                self.send_response(status)
                self._send_common_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_common_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", config.cors_allow_origin)
                self.send_header(
                    "Access-Control-Allow-Headers",
                    "Authorization, Content-Type, x-api-key, anthropic-version",
                )
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

            def _safe_write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
                try:
                    self._write_json(status, payload)
                except CLIENT_DISCONNECTED:
                    logger.warning("客户端在 JSON 响应写回前断开 path=%s", self.path)

            def _parse_content_length(self) -> int:
                raw_value = self.headers.get("Content-Length", "0").strip()
                try:
                    return int(raw_value or "0")
                except ValueError as exc:
                    raise ValueError(f"无效的 Content-Length: {raw_value}") from exc

            def _write_sse_error(self, message: str, error_type: str) -> None:
                write_sse_error(
                    SSEWriter(self.wfile),
                    message,
                    error_type,
                    logger=logger,
                    path=self.path,
                )

            def _write_anthropic_sse_error(self, message: str, error_type: str) -> None:
                write_anthropic_sse_error(
                    SSEWriter(self.wfile),
                    message,
                    error_type,
                    logger=logger,
                    path=self.path,
                )

            def _write_responses_sse_error(
                self,
                accumulator: OpenAIResponsesStreamAccumulator,
                message: str,
                error_type: str,
            ) -> None:
                write_responses_sse_error(
                    SSEWriter(self.wfile),
                    accumulator,
                    message,
                    error_type,
                    logger=logger,
                    path=self.path,
                )

            def _debug_log_request_start(self) -> None:
                debug_dump(
                    logger,
                    config.debug_dump_all,
                    f"HTTP 入站请求 {self.command} {self.path} headers",
                    {key: value for key, value in self.headers.items()},
                )

            def _path_without_query(self) -> str:
                return urlparse(self.path).path

            def log_message(self, format: str, *args) -> None:
                logger.info("%s - %s", self.address_string(), format % args)

        return RequestHandler
