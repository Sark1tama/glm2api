"""OpenAI Video vertical slice for ChatGLM's asynchronous video flow."""

from __future__ import annotations

import json
import mimetypes
import re
import struct
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from email.generator import _make_boundary  # type: ignore

from ..config import DEFAULT_VIDEO_MODEL_NAME
from ..glm.errors import UpstreamAPIError
from ..infrastructure.logging import debug_dump

if TYPE_CHECKING:
    from ..glm.client import GLMWebClient


VIDEO_MAX_PROMPT_LENGTH = 32_000
VIDEO_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
VIDEO_POLL_INTERVAL_SECONDS = 3.0
VIDEO_TIMEOUT_SECONDS = 900.0
VIDEO_JOB_RETENTION_SECONDS = 24 * 60 * 60
VIDEO_MAX_STORED_JOBS = 1000


class VideoNotFoundError(LookupError):
    pass


class VideoNotReadyError(RuntimeError):
    pass


class _TrustedVideoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate video content redirects before forwarding GLM credentials."""

    def __init__(self, normalize_url):
        self._normalize_url = normalize_url

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        normalized = self._normalize_url(newurl)
        if not normalized:
            raise UpstreamAPIError(status_code=502, message="视频重定向地址无效。")
        return super().redirect_request(req, fp, code, msg, headers, normalized)


@dataclass(frozen=True, slots=True)
class VideoInputReference:
    url: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    content: bytes | None = None


@dataclass(frozen=True, slots=True)
class NormalizedVideoRequest:
    model: str
    prompt: str
    seconds: str
    size: str
    ratio_width: int
    ratio_height: int
    generation_pattern: int
    resolution: int
    fps: int
    generation_ai_audio: int
    label_watermark: int
    input_reference: VideoInputReference | None = None


@dataclass(slots=True)
class VideoJob:
    id: str
    model: str
    prompt: str
    seconds: str
    size: str
    created_at: int
    status: str = "queued"
    progress: int = 0
    completed_at: int | None = None
    error: dict[str, object] | None = None
    content_url: str | None = None
    poster_url: str | None = None
    upstream_id: str | None = None
    account_index: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "id": self.id,
                "object": "video",
                "model": self.model,
                "status": self.status,
                "progress": self.progress,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
                "expires_at": None,
                "prompt": self.prompt,
                "seconds": self.seconds,
                "size": self.size,
                "error": self.error,
            }

    def set_in_progress(self) -> None:
        with self._lock:
            self.status = "in_progress"

    def set_upstream(self, upstream_id: str, account_index: int | None) -> None:
        with self._lock:
            self.upstream_id = upstream_id
            self.account_index = account_index

    def set_progress(self, progress: int) -> None:
        with self._lock:
            self.progress = max(0, min(progress, 99))

    def complete(self, content_url: str, poster_url: str | None = None) -> None:
        with self._lock:
            self.status = "completed"
            self.progress = 100
            self.completed_at = int(time.time())
            self.content_url = content_url
            self.poster_url = poster_url

    def fail(self, message: str, error_type: str = "upstream_error") -> None:
        with self._lock:
            self.status = "failed"
            self.error = {"message": message, "type": error_type}


class VideoJobStore:
    def __init__(
        self,
        *,
        max_jobs: int = VIDEO_MAX_STORED_JOBS,
        retention_seconds: int = VIDEO_JOB_RETENTION_SECONDS,
    ) -> None:
        if max_jobs <= 0:
            raise ValueError("视频任务存储上限必须大于 0。")
        if retention_seconds <= 0:
            raise ValueError("视频任务保留时间必须大于 0。")
        self.max_jobs = max_jobs
        self.retention_seconds = retention_seconds
        self._jobs: dict[str, VideoJob] = {}
        self._lock = threading.Lock()

    def add(self, job: VideoJob) -> None:
        with self._lock:
            self._prune_locked()
            self._jobs[job.id] = job
            self._prune_locked()

    def get(self, video_id: str) -> VideoJob:
        with self._lock:
            self._prune_locked()
            job = self._jobs.get(video_id)
        if job is None:
            raise VideoNotFoundError(f"视频任务不存在: {video_id}")
        return job

    def list(self) -> list[VideoJob]:
        with self._lock:
            self._prune_locked()
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def _prune_locked(self) -> None:
        now = int(time.time())
        terminal_statuses = {"completed", "failed"}
        for job_id, job in list(self._jobs.items()):
            with job._lock:
                status = job.status
                finished_at = job.completed_at
                created_at = job.created_at
            if status in terminal_statuses and now - (finished_at or created_at) >= self.retention_seconds:
                self._jobs.pop(job_id, None)

        if len(self._jobs) <= self.max_jobs:
            return
        terminal_jobs = []
        for job in self._jobs.values():
            with job._lock:
                if job.status in terminal_statuses:
                    terminal_jobs.append((job.completed_at or job.created_at, job.id))
        for _, job_id in sorted(terminal_jobs):
            if len(self._jobs) <= self.max_jobs:
                break
            self._jobs.pop(job_id, None)


def normalize_video_request(payload: dict[str, object]) -> NormalizedVideoRequest:
    if not isinstance(payload, dict):
        raise ValueError("视频请求体顶层必须是 JSON 对象或 multipart 表单。")

    prompt = str(payload.get("prompt", "")).strip()
    input_reference = _parse_input_reference(payload)
    if not prompt and input_reference is None:
        raise ValueError("视频生成请求必须包含 prompt，或提供 input_reference。")
    if len(prompt) > VIDEO_MAX_PROMPT_LENGTH:
        raise ValueError(f"prompt 不能超过 {VIDEO_MAX_PROMPT_LENGTH} 个字符。")

    model = str(payload.get("model", DEFAULT_VIDEO_MODEL_NAME)).strip() or DEFAULT_VIDEO_MODEL_NAME
    seconds = _parse_choice(payload.get("seconds", "5"), {"5", "10"}, "seconds")
    size, ratio_width, ratio_height = _parse_size(payload.get("size", "1280x720"))

    quality = str(payload.get("quality", "speed")).strip().lower()
    generation_pattern = {"speed": 1, "fast": 1, "quality": 2}.get(quality)
    if generation_pattern is None:
        raise ValueError("quality 只支持 speed 或 quality。")

    resolution_name = str(payload.get("resolution", "720p")).strip().lower()
    resolution = {"720p": 2, "1080p": 0, "4k": 1}.get(resolution_name)
    if resolution is None:
        raise ValueError("resolution 只支持 720p、1080p 或 4k。")

    fps_value = payload.get("fps", 30)
    try:
        fps = int(fps_value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("fps 只支持 30 或 60。") from exc
    if fps not in {30, 60}:
        raise ValueError("fps 只支持 30 或 60。")

    return NormalizedVideoRequest(
        model=model,
        prompt=prompt,
        seconds=seconds,
        size=size,
        ratio_width=ratio_width,
        ratio_height=ratio_height,
        generation_pattern=generation_pattern,
        resolution=resolution,
        fps=0 if fps == 30 else 1,
        generation_ai_audio=1 if _parse_bool(payload.get("generate_audio", True)) else 0,
        label_watermark=1 if _parse_bool(payload.get("remove_watermark", True)) else 0,
        input_reference=input_reference,
    )


def build_upstream_video_payload(
    request: NormalizedVideoRequest,
    source_ids: list[str] | tuple[str, ...] = (),
    ratio: tuple[int, int] | None = None,
) -> dict[str, object]:
    ratio_width, ratio_height = ratio or (request.ratio_width, request.ratio_height)
    duration = 1 if request.seconds == "5" else 2
    if len(source_ids) == 2:
        duration = 1

    base_parameter_extra: dict[str, object] = {
        "generation_pattern": request.generation_pattern,
        "resolution": request.resolution,
        "fps": request.fps,
        "duration": duration,
        "generation_ai_audio": request.generation_ai_audio,
        "generation_ratio_width": ratio_width,
        "generation_ratio_height": ratio_height,
        "activity_type": 0,
        "label_watermark": request.label_watermark,
        "generation_type": "reference_img" if source_ids else "",
        "prompt": request.prompt,
    }
    payload: dict[str, object] = {
        "prompt": request.prompt,
        "conversation_id": "",
        "base_parameter_extra": base_parameter_extra,
    }
    if source_ids:
        payload["source_list"] = list(source_ids)
    else:
        payload["advanced_parameter_extra"] = {}
    return payload


def _parse_input_reference(payload: dict[str, object]) -> VideoInputReference | None:
    raw_file = payload.get("_input_reference_file")
    if isinstance(raw_file, dict):
        content = raw_file.get("content")
        if not isinstance(content, bytes):
            raise ValueError("input_reference 文件内容无效。")
        if len(content) > VIDEO_MAX_UPLOAD_BYTES:
            raise ValueError("input_reference 文件超过 100MB。")
        return VideoInputReference(
            filename=str(raw_file.get("filename") or "reference-image"),
            mime_type=str(raw_file.get("mime_type") or "application/octet-stream"),
            content=content,
        )

    raw = payload.get("input_reference")
    if raw is None or raw == "":
        raw = payload.get("file")
    if raw is None or raw == "":
        return None
    if isinstance(raw, str) and raw.lstrip().startswith("{"):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("input_reference JSON 无效。") from exc
    if isinstance(raw, dict):
        image_url = raw.get("image_url", raw.get("url"))
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str) or not image_url.strip():
            raise ValueError("input_reference 目前只支持 image_url/url。")
        raw = image_url
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("input_reference 必须是图片 URL 或 data URL。")
    return VideoInputReference(url=raw.strip())


def _parse_size(value: object) -> tuple[str, int, int]:
    size = str(value).strip().lower()
    match = re.fullmatch(r"([1-9][0-9]{0,4})x([1-9][0-9]{0,4})", size)
    if match is None:
        raise ValueError("size 必须是类似 1280x720 的宽高格式。")
    width, height = int(match.group(1)), int(match.group(2))
    divisor = _gcd(width, height)
    return size, width // divisor, height // divisor


def _parse_choice(value: object, choices: set[str], name: str) -> str:
    normalized = str(value).strip()
    if normalized not in choices:
        allowed = "、".join(sorted(choices, key=int))
        raise ValueError(f"{name} 只支持 {allowed}。")
    return normalized


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _gcd(left: int, right: int) -> int:
    while right:
        left, right = right, left % right
    return max(left, 1)


def new_video_job(request: NormalizedVideoRequest) -> VideoJob:
    return VideoJob(
        id=f"video_{uuid.uuid4().hex}",
        model=request.model,
        prompt=request.prompt,
        seconds=request.seconds,
        size=request.size,
        created_at=int(time.time()),
    )


class VideoService:
    """Own the OpenAI Video to ChatGLM asynchronous task flow."""

    def __init__(self, client: GLMWebClient) -> None:
        self.client = client
        self.jobs = VideoJobStore()

    def create(self, request: NormalizedVideoRequest) -> dict[str, object]:
        lease = self.client.request_queue.acquire(f"video:{request.model}")
        job = new_video_job(request)
        self.jobs.add(job)
        try:
            threading.Thread(
                target=self._run_video_job,
                args=(job, request, lease),
                name=f"glm-video-{job.id}",
                daemon=True,
            ).start()
        except Exception:
            lease.release()
            job.fail("视频任务启动失败。", error_type="internal_error")
            raise

        self.client.logger.info("视频任务已创建 id=%s model=%s seconds=%s size=%s", job.id, request.model, request.seconds, request.size)
        return job.snapshot()

    def retrieve(self, video_id: str) -> dict[str, object]:
        return self.jobs.get(video_id).snapshot()

    def list(self) -> dict[str, object]:
        return {
            "object": "list",
            "data": [job.snapshot() for job in self.jobs.list()],
        }

    def open_content(self, video_id: str):
        job = self.jobs.get(video_id)
        with job._lock:
            status = job.status
            content_url = job.content_url
            account_index = job.account_index
        if status != "completed" or not content_url:
            raise VideoNotReadyError(f"视频任务尚未完成，当前状态为 {status}。")
        content_url = self._normalize_video_url(content_url)
        if not content_url:
            raise UpstreamAPIError(status_code=502, message="视频任务缺少受信任的结果 URL。")

        def send_request(account_index: int, access_token: str):
            request = self.client.build_signed_request(
                content_url,
                method="GET",
                access_token=access_token,
                app_fr="default",
                referer="https://chatglm.cn/video?lang=zh",
            )
            opener = urllib.request.build_opener(
                _TrustedVideoRedirectHandler(self._normalize_video_url)
            )
            return self.client.open_upstream_request(request, opener=opener)

        return self.client.call_with_account_failover(
            "video_content",
            send_request,
            preferred_account_index=account_index,
        )

    def _run_video_job(self, job: VideoJob, request: NormalizedVideoRequest, lease: QueueLease) -> None:
        try:
            job.set_in_progress()
            source_ids: list[str] = []
            ratio: tuple[int, int] | None = None
            if request.input_reference is not None:
                source_id, account_index, ratio = self._upload_video_reference(
                    request.input_reference,
                    fallback_ratio=(request.ratio_width, request.ratio_height),
                    preferred_account_index=self.client.get_preferred_account_index(lease.ticket),
                )
                source_ids.append(source_id)
                job.account_index = account_index

            upstream_payload = build_upstream_video_payload(request, source_ids=source_ids, ratio=ratio)
            response_payload, account_index = self._video_json_request(
                "POST",
                self.client.config.video_chat_url,
                upstream_payload,
                preferred_account_index=(
                    job.account_index
                    if job.account_index is not None
                    else self.client.get_preferred_account_index(lease.ticket)
                ),
                request_name="video_create",
            )
            upstream_id = self._extract_video_id(response_payload)
            if not upstream_id:
                raise UpstreamAPIError(
                    status_code=502,
                    message="GLM 视频创建成功响应中没有 chat_id。",
                    payload=response_payload,
                )
            job.set_upstream(upstream_id, account_index)

            deadline = time.monotonic() + VIDEO_TIMEOUT_SECONDS
            while True:
                status_payload, account_index = self._video_json_request(
                    "GET",
                    f"{self.client.config.video_api_base_url}/chat/status/{urllib.parse.quote(upstream_id, safe='')}",
                    None,
                    preferred_account_index=account_index,
                    request_name="video_status",
                )
                job.set_upstream(upstream_id, account_index)
                state = self._extract_video_state(status_payload)
                if state["error"]:
                    raise UpstreamAPIError(status_code=502, message=str(state["error"]), payload=status_payload)
                if state["progress"] is not None:
                    job.set_progress(state["progress"])
                if state["completed"] or state["video_url"]:
                    detail_payload, account_index = self._video_json_request(
                        "GET",
                        f"{self.client.config.video_api_base_url}/chat/{urllib.parse.quote(upstream_id, safe='')}",
                        None,
                        preferred_account_index=account_index,
                        request_name="video_detail",
                    )
                    job.set_upstream(upstream_id, account_index)
                    video_url = self._extract_video_url(detail_payload) or state["video_url"]
                    if not video_url:
                        raise UpstreamAPIError(
                            status_code=502,
                            message="GLM 视频任务已完成，但详情中没有可下载的视频地址。",
                            payload=detail_payload,
                        )
                    normalized_video_url = self._normalize_video_url(video_url)
                    normalized_poster_url = self._normalize_video_url(self._extract_video_poster(detail_payload))
                    job.complete(normalized_video_url, normalized_poster_url)
                    self.client.logger.info("视频任务完成 id=%s upstream_id=%s", job.id, upstream_id)
                    return
                if state["failed"]:
                    raise UpstreamAPIError(
                        status_code=502,
                        message=state["error"] or "GLM 视频生成失败。",
                        payload=status_payload,
                    )
                if time.monotonic() >= deadline:
                    raise UpstreamAPIError(status_code=504, message="GLM 视频生成超时。", payload={"upstream_id": upstream_id})
                time.sleep(VIDEO_POLL_INTERVAL_SECONDS)
        except Exception as exc:
            job.fail(str(exc))
            self.client.logger.warning("视频任务失败 id=%s error=%s", job.id, exc)
        finally:
            lease.release()

    def _upload_video_reference(
        self,
        reference: VideoInputReference,
        fallback_ratio: tuple[int, int],
        preferred_account_index: int | None,
    ) -> tuple[str, int | None, tuple[int, int]]:
        if reference.content is not None:
            filename = reference.filename or f"reference-{uuid.uuid4().hex}.png"
            mime_type = reference.mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
            payload = reference.content
        elif reference.url:
            filename, mime_type, payload = self.client.files.fetch_file_payload(reference.url)
        else:
            raise ValueError("input_reference 缺少图片内容。")
        if not mime_type.startswith("image/"):
            raise ValueError(f"input_reference 必须是图片文件，实际类型为 {mime_type}。")
        if len(payload) > VIDEO_MAX_UPLOAD_BYTES:
            raise ValueError("input_reference 文件超过 100MB，拒绝上传。")
        filename = filename.replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")

        dimensions = self._image_dimensions(payload, mime_type)
        ratio = self._ratio_from_dimensions(dimensions) if dimensions else fallback_ratio
        width, height = dimensions or fallback_ratio
        boundary = _make_boundary()
        body = self._build_video_upload_multipart(boundary, filename, mime_type, payload, width, height)

        selected_account = preferred_account_index

        def send_request(account_index: int, access_token: str):
            nonlocal selected_account
            selected_account = account_index
            request = self.client.build_signed_request(
                self.client.config.video_upload_url,
                method="POST",
                access_token=access_token,
                data=body,
                app_fr="default",
                referer="https://chatglm.cn/video?lang=zh",
                extra_headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            debug_dump(
                self.client.logger,
                self.client.config.debug_dump_all,
                f"转发到 GLM 的 video upload 请求头 account={account_index}",
                dict(request.header_items()),
            )
            self.client.logger.debug(
                "准备上传视频参考图 filename=%s mime=%s bytes=%s width=%s height=%s",
                filename,
                mime_type,
                len(payload),
                width,
                height,
            )
            with self.client.open_upstream_request(request) as response:
                result = self.client.auth.read_json_response(response)
            self._raise_for_video_payload(result)
            return result

        response_payload = self.client.call_with_account_failover(
            "video_upload",
            send_request,
            preferred_account_index=preferred_account_index,
        )
        result = response_payload.get("result", response_payload)
        if not isinstance(result, dict):
            raise UpstreamAPIError(status_code=502, message="GLM 视频参考图上传响应格式异常。", payload=response_payload)
        source_id = str(result.get("source_id", "")).strip()
        if not source_id:
            raise UpstreamAPIError(status_code=502, message="GLM 视频参考图上传响应中没有 source_id。", payload=response_payload)
        return source_id, selected_account, ratio

    def _video_json_request(
        self,
        method: str,
        url: str,
        payload: dict[str, object] | None,
        preferred_account_index: int | None,
        request_name: str,
    ) -> tuple[dict[str, object], int | None]:
        request_body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") if payload is not None else None
        selected_account: int | None = preferred_account_index

        def send_request(account_index: int, access_token: str):
            nonlocal selected_account
            selected_account = account_index
            request = self.client.build_signed_request(
                url,
                method=method,
                access_token=access_token,
                data=request_body,
                app_fr="default",
                referer="https://chatglm.cn/video?lang=zh",
            )
            debug_dump(
                self.client.logger,
                self.client.config.debug_dump_all,
                f"转发到 GLM 的 {request_name} 请求头 account={account_index}",
                dict(request.header_items()),
            )
            debug_dump(self.client.logger, self.client.config.debug_dump_all, f"转发到 GLM 的 {request_name} 请求体", payload or {})
            with self.client.open_upstream_request(request) as response:
                result = self.client.auth.read_json_response(response)
            self._raise_for_video_payload(result)
            return result

        result = self.client.call_with_account_failover(
            request_name,
            send_request,
            preferred_account_index=preferred_account_index,
        )
        return result, selected_account

    def _build_video_upload_multipart(
        self,
        boundary: str,
        filename: str,
        mime_type: str,
        payload: bytes,
        width: int,
        height: int,
    ) -> bytes:
        parts = [
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
            + payload,
            f'--{boundary}\r\nContent-Disposition: form-data; name="width"\r\n\r\n{width}'.encode("utf-8"),
            f'--{boundary}\r\nContent-Disposition: form-data; name="height"\r\n\r\n{height}'.encode("utf-8"),
        ]
        return b"\r\n".join(parts) + f"\r\n--{boundary}--\r\n".encode("utf-8")

    def _raise_for_video_payload(self, payload: dict[str, object]) -> None:
        code = payload.get("code")
        status = payload.get("status")
        numeric_failure = isinstance(code, (int, float)) and code != 0
        numeric_status_failure = isinstance(status, (int, float)) and status != 0
        textual_failure = isinstance(status, str) and status.strip().lower() in {"error", "failed", "failure"}
        if numeric_failure or numeric_status_failure or textual_failure:
            raise UpstreamAPIError(
                status_code=502,
                message=self.client.build_error_message(200, payload),
                payload=payload,
            )

    def _extract_video_id(self, payload: dict[str, object]) -> str | None:
        for container in self._video_containers(payload):
            for key in ("chat_id", "task_id", "video_id", "id"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _extract_video_state(self, payload: dict[str, object]) -> dict[str, object]:
        status: str | None = None
        progress: int | None = None
        error: str | None = None
        video_url = self._extract_video_url(payload)
        for container in self._video_containers(payload):
            if status is None:
                for key in ("status", "state", "chat_status"):
                    value = container.get(key)
                    if isinstance(value, str) and value.strip():
                        status = value.strip().lower()
                        break
            if progress is None:
                for key in ("progress", "percent", "percentage"):
                    value = container.get(key)
                    if isinstance(value, (int, float)):
                        progress = int(value * 100 if 0 <= value <= 1 else value)
                        break
            if error is None:
                for key in ("error", "err_msg", "error_message", "message"):
                    value = container.get(key)
                    if isinstance(value, str) and value.strip() and key != "message":
                        error = value.strip()
                        break
        completed = status in {"success", "succeeded", "completed", "complete", "finished", "finish", "done"}
        failed = status in {"error", "failed", "failure", "cancelled", "canceled"}
        return {
            "completed": completed,
            "failed": failed,
            "progress": progress,
            "error": error,
            "video_url": video_url,
        }

    def _video_containers(self, payload: dict[str, object]) -> list[dict[str, object]]:
        containers: list[dict[str, object]] = []
        for candidate in (payload.get("result"), payload.get("data"), payload):
            if isinstance(candidate, dict) and candidate not in containers:
                containers.append(candidate)
        return containers

    def _extract_video_url(self, payload: dict[str, object], allow_generic: bool = False) -> str | None:
        for key in ("output",):
            nested = payload.get(key)
            if isinstance(nested, dict):
                url = self._extract_video_url(nested, allow_generic=True)
                if url:
                    return url
        for key in ("result", "data"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                url = self._extract_video_url(nested, allow_generic=False)
                if url:
                    return url
        for key in ("video_url", "videoUrl", "file_url"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if allow_generic:
            value = payload.get("url")
            if isinstance(value, str) and value.strip():
                return value
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        url = self._extract_video_url(item, allow_generic=allow_generic)
                        if url:
                            return url
        return None

    def _extract_video_poster(self, payload: dict[str, object]) -> str | None:
        for key in ("cover_url", "poster_url", "cover", "poster"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for key in ("output", "result", "data"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                poster = self._extract_video_poster(nested)
                if poster:
                    return poster
        return None

    def _normalize_video_url(self, value: str | None) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip()
        base = urllib.parse.urlparse(self.client.config.glm_base_url)
        if not base.scheme or not base.netloc:
            raise UpstreamAPIError(status_code=502, message="GLM 基础地址无效，无法打开视频结果。")
        origin = f"{base.scheme}://{base.netloc}"
        normalized = urllib.parse.urljoin(origin, value)
        parsed = urllib.parse.urlparse(normalized)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        trusted_base_host = (base.hostname or "").lower().rstrip(".")
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or not hostname
            or not (
                hostname == trusted_base_host
                or hostname == "chatglm.cn"
                or hostname.endswith(".chatglm.cn")
            )
        ):
            raise UpstreamAPIError(status_code=502, message="GLM 返回了不受信任的视频结果 URL。")
        return normalized

    def _image_dimensions(self, payload: bytes, mime_type: str) -> tuple[int, int] | None:
        if payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 24:
            return struct.unpack(">II", payload[16:24])
        if payload[:3] == b"GIF" and len(payload) >= 10:
            return struct.unpack("<HH", payload[6:10])
        if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
            if payload[12:16] == b"VP8X" and len(payload) >= 30:
                width = int.from_bytes(payload[24:27], "little") + 1
                height = int.from_bytes(payload[27:30], "little") + 1
                return width, height
        if payload[:2] == b"\xff\xd8":
            index = 2
            while index + 9 < len(payload):
                if payload[index] != 0xFF:
                    index += 1
                    continue
                while index < len(payload) and payload[index] == 0xFF:
                    index += 1
                if index >= len(payload):
                    break
                marker = payload[index]
                index += 1
                if marker in {0xD8, 0xD9}:
                    continue
                if index + 2 > len(payload):
                    break
                segment_length = struct.unpack(">H", payload[index:index + 2])[0]
                if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                    if index + 7 <= len(payload):
                        height, width = struct.unpack(">HH", payload[index + 3:index + 7])
                        return width, height
                if segment_length < 2:
                    break
                index += segment_length
        return None

    def _ratio_from_dimensions(self, dimensions: tuple[int, int] | None) -> tuple[int, int] | None:
        if not dimensions:
            return None
        width, height = dimensions
        divisor = self._gcd(width, height)
        return width // divisor, height // divisor

    def _gcd(self, left: int, right: int) -> int:
        while right:
            left, right = right, left % right
        return max(left, 1)


__all__ = [
    "NormalizedVideoRequest",
    "VideoInputReference",
    "VideoJob",
    "VideoJobStore",
    "VideoNotFoundError",
    "VideoNotReadyError",
    "VideoService",
    "VIDEO_JOB_RETENTION_SECONDS",
    "VIDEO_MAX_STORED_JOBS",
    "build_upstream_video_payload",
    "new_video_job",
    "normalize_video_request",
]
