"""OpenAI Images vertical slice for the ChatGLM image flow."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import re
import socket
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import BUILTIN_IMAGE_MODELS, DEFAULT_CHAT_MODEL_NAME
from ..glm.errors import UpstreamAPIError
from ..glm.events import effective_event_status, is_nonzero_status
from ..infrastructure.logging import debug_dump

if TYPE_CHECKING:
    from ..glm.client import GLMWebClient


SUPPORTED_IMAGE_RESPONSE_FORMATS = frozenset({"url", "b64_json"})
MAX_IMAGE_COUNT = 10
IMAGE_SIZE_TO_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1024x1536": "2:3",
    "1536x1024": "3:2",
    "1024x1792": "9:16",
    "1792x1024": "16:9",
}
IMAGE_TERMINAL_STATUSES = frozenset({"finish", "finished", "done", "complete", "completed", "intervene"})
IMAGE_DOWNLOAD_MAX_BYTES = 100 * 1024 * 1024
_MAX_DATA_URL_LENGTH = 4 * ((IMAGE_DOWNLOAD_MAX_BYTES + 2) // 3)
_TRUSTED_IMAGE_HOSTS = frozenset({"sfile.chatglm.cn"})


class ImageResponseConversionError(ValueError):
    """The upstream result cannot be represented by the requested image API."""


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """A generated image URL or inline base64 asset."""

    url: str | None = None
    base64_data: str | None = None
    revised_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.url and not self.base64_data:
            raise ValueError("generated image requires a URL or base64 data")


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest:
    """Normalized OpenAI Images request used by the image service."""

    model: str
    prompt: str
    size: str = "1024x1024"
    n: int = 1
    response_format: str = "url"
    style: str = "none"
    scene: str = "none"
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    """Completed image assets from the ChatGLM stream."""

    created: int
    images: tuple[GeneratedImage, ...]


def _validate_public_image_url(image_url: str) -> None:
    """Reject non-HTTP or non-public image destinations before connecting."""
    parsed = urllib.parse.urlparse(image_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise UpstreamAPIError(status_code=502, message="生成图片 URL 协议不受支持。")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if (
        not hostname
        or hostname in {"localhost", "localhost.localdomain"}
        or hostname.endswith(".local")
        or hostname.endswith(".localhost")
    ):
        raise UpstreamAPIError(status_code=502, message="生成图片 URL 指向本地地址。")

    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        literal_address = None
    if literal_address is not None:
        if not literal_address.is_global:
            raise UpstreamAPIError(status_code=502, message="生成图片 URL 指向非公网地址。")
        return

    if hostname in _TRUSTED_IMAGE_HOSTS:
        return

    try:
        address_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UpstreamAPIError(status_code=502, message="生成图片 URL 无法解析。") from exc

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for info in address_infos:
        try:
            addresses.add(ipaddress.ip_address(info[4][0]))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise UpstreamAPIError(status_code=502, message="生成图片 URL 解析结果无效。") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise UpstreamAPIError(status_code=502, message="生成图片 URL 指向非公网地址。")


class _PublicImageRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate every redirect target before urllib follows it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def openai_images_to_internal(
    payload: Mapping[str, object],
    *,
    default_model: str,
) -> ImageGenerationRequest:
    """Convert an OpenAI Images request into the image-local request model."""
    if not isinstance(payload, Mapping):
        raise ValueError("图片生成请求体顶层必须是 JSON 对象。")

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("图片生成请求必须包含 prompt 字段。")

    raw_model = payload.get("model")
    model = str(raw_model or default_model).strip() or default_model
    if model not in BUILTIN_IMAGE_MODELS:
        supported = ", ".join(BUILTIN_IMAGE_MODELS)
        raise ValueError(f"当前图片接口不支持模型 {model}；支持的模型：{supported}。")

    raw_n = payload.get("n", 1)
    if isinstance(raw_n, bool) or not isinstance(raw_n, int):
        raise ValueError("图片生成请求的 n 必须是整数。")
    if not 1 <= raw_n <= MAX_IMAGE_COUNT:
        raise ValueError(f"图片生成请求的 n 必须在 1 到 {MAX_IMAGE_COUNT} 之间。")

    response_format = str(payload.get("response_format") or "url").strip().lower()
    if response_format not in SUPPORTED_IMAGE_RESPONSE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_RESPONSE_FORMATS))
        raise ValueError(f"图片生成请求的 response_format 只支持: {supported}。")

    known_fields = {"model", "prompt", "size", "n", "response_format", "style", "scene"}
    return ImageGenerationRequest(
        model=model,
        prompt=prompt,
        size=str(payload.get("size") or "1024x1024"),
        n=raw_n,
        response_format=response_format,
        style=str(payload.get("style") or "none"),
        scene=str(payload.get("scene") or "none"),
        extra={key: value for key, value in payload.items() if key not in known_fields},
    )


def internal_to_openai_images_response(
    request: ImageGenerationRequest,
    result: ImageGenerationResult,
    *,
    download_image: Callable[[str], str],
) -> dict[str, object]:
    """Serialize a completed image result at the OpenAI HTTP boundary."""
    response_format = request.response_format.strip().lower()
    if response_format not in SUPPORTED_IMAGE_RESPONSE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_RESPONSE_FORMATS))
        raise ValueError(f"图片生成请求的 response_format 只支持: {supported}。")
    if isinstance(request.n, bool) or not 1 <= request.n <= MAX_IMAGE_COUNT:
        raise ValueError(f"图片生成请求的 n 必须在 1 到 {MAX_IMAGE_COUNT} 之间。")

    data: list[dict[str, object]] = []
    for image in result.images[: request.n]:
        item: dict[str, object] = {}
        if response_format == "b64_json":
            if image.base64_data:
                item["b64_json"] = image.base64_data
            elif image.url:
                item["b64_json"] = download_image(image.url)
            else:
                continue
        elif image.url:
            item["url"] = image.url
        else:
            continue

        if image.revised_prompt:
            item["revised_prompt"] = image.revised_prompt
        data.append(item)

    if not data:
        raise ImageResponseConversionError("图片结果中没有可转换的图片数据。")
    return {"created": result.created, "data": data}


def build_glm_image_payload(
    request: ImageGenerationRequest,
    *,
    assistant_id: str,
    selected_model: str = DEFAULT_CHAT_MODEL_NAME,
) -> dict[str, object]:
    """Build the private ChatGLM web payload for an image request."""
    return {
        "assistant_id": assistant_id,
        "conversation_id": "",
        "project_id": "",
        "chat_type": "user_chat",
        "meta_data": {
            "cogview": {
                "aspect_ratio": resolve_image_aspect_ratio(request.size),
                "style": request.style.strip().lower() or "none",
                "scene": request.scene.strip().lower() or "none",
                "chat_model": "",
                "rm_label_watermark": False,
            },
            "is_test": False,
            "input_question_type": "xxxx",
            "channel": "",
            "draft_id": "",
            "chat_mode": "",
            "is_networking": False,
            "quote_log_id": "",
            "platform": "pc",
            "selected_model": selected_model,
        },
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": request.prompt.strip()}],
            }
        ],
    }


def resolve_image_aspect_ratio(size: str) -> str:
    normalized = size.strip().lower()
    if normalized in IMAGE_SIZE_TO_ASPECT_RATIO:
        return IMAGE_SIZE_TO_ASPECT_RATIO[normalized]
    if re.fullmatch(r"\d+x\d+", normalized):
        width_str, height_str = normalized.split("x", 1)
        width = max(int(width_str), 1)
        height = max(int(height_str), 1)
        return f"{width}:{height}"
    return "1:1"


@dataclass(slots=True)
class GLMImageEventAccumulator:
    """Collect image parts without applying text-stream rendering rules."""

    conversation_id: str = ""
    _parts_by_logic_id: dict[str, dict[str, object]] = field(default_factory=dict, repr=False)
    _part_order: list[str] = field(default_factory=list, repr=False)
    _unkeyed_parts: list[dict[str, object]] = field(default_factory=list, repr=False)
    _direct_images: list[GeneratedImage] = field(default_factory=list, repr=False)
    _last_status: str | None = field(default=None, repr=False)

    def consume_event(self, payload: Mapping[str, object]) -> str | None:
        """Consume one ChatGLM event and return its normalized status."""
        if not self.conversation_id and payload.get("conversation_id"):
            self.conversation_id = str(payload["conversation_id"])

        status = _normalize_status(payload.get("status"))
        if status:
            self._last_status = status

        raw_parts = payload.get("parts")
        if isinstance(raw_parts, list) and raw_parts:
            for raw_part in raw_parts:
                if not isinstance(raw_part, dict):
                    continue
                logic_id = str(raw_part.get("logic_id") or "").strip()
                if logic_id:
                    if logic_id not in self._parts_by_logic_id:
                        self._part_order.append(logic_id)
                    self._parts_by_logic_id[logic_id] = raw_part
                else:
                    self._unkeyed_parts.append(raw_part)
        else:
            self._direct_images.extend(_extract_direct_images(payload))
        return status

    def build_result(self, *, created: int | None = None) -> ImageGenerationResult:
        """Build the protocol-neutral result in upstream arrival order."""
        images: list[GeneratedImage] = list(self._direct_images)
        terminal_event = self._last_status in IMAGE_TERMINAL_STATUSES
        parts = [self._parts_by_logic_id[logic_id] for logic_id in self._part_order]
        parts.extend(self._unkeyed_parts)
        for part in parts:
            part_status = _normalize_status(part.get("status"))
            if part_status and part_status not in IMAGE_TERMINAL_STATUSES:
                continue
            if not part_status and not terminal_event:
                continue
            images.extend(_extract_part_images(part))

        deduplicated: list[GeneratedImage] = []
        seen: set[tuple[str, str]] = set()
        for image in images:
            identity = (image.url or "", image.base64_data or "")
            if identity in seen:
                continue
            seen.add(identity)
            deduplicated.append(image)
        return ImageGenerationResult(
            created=int(time.time()) if created is None else created,
            images=tuple(deduplicated),
        )


def _normalize_status(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _extract_part_images(part: Mapping[str, object]) -> list[GeneratedImage]:
    content = part.get("content")
    if isinstance(content, dict):
        content_items = [content]
    elif isinstance(content, list):
        content_items = [item for item in content if isinstance(item, dict)]
    else:
        content_items = []

    images: list[GeneratedImage] = []
    for content_item in content_items:
        if str(content_item.get("type", "")).strip().lower() != "image":
            continue
        revised_prompt = _revised_prompt(content_item)
        raw_images = content_item.get("image")
        if isinstance(raw_images, dict):
            raw_images = [raw_images]
        if isinstance(raw_images, list):
            for raw_image in raw_images:
                image = _image_from_mapping(raw_image, revised_prompt=revised_prompt)
                if image is not None:
                    images.append(image)
    return images


def _extract_direct_images(payload: Mapping[str, object]) -> list[GeneratedImage]:
    """Accept common JSON image-result envelopes as well as SSE-shaped data."""
    if str(payload.get("type", "")).strip().lower() == "image":
        return _extract_part_images({"content": [payload]})

    images: list[GeneratedImage] = []
    for key in ("data", "images", "image", "result", "output", "content"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            image = _image_from_mapping(candidate)
            if image is not None:
                images.append(image)
            images.extend(_extract_direct_images(candidate))
        elif isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, dict):
                    image = _image_from_mapping(item)
                    if image is not None:
                        images.append(image)
                    images.extend(_extract_direct_images(item))
    return images


def _image_from_mapping(value: object, *, revised_prompt: str | None = None) -> GeneratedImage | None:
    if not isinstance(value, dict):
        return None
    raw_url = value.get("image_url", value.get("url"))
    if isinstance(raw_url, dict):
        raw_url = raw_url.get("url")
    url = str(raw_url).strip() if isinstance(raw_url, str) and raw_url.strip() else None
    raw_base64 = value.get("b64_json", value.get("base64_data"))
    base64_data = str(raw_base64).strip() if isinstance(raw_base64, str) and raw_base64.strip() else None
    if not url and not base64_data:
        return None
    return GeneratedImage(url=url, base64_data=base64_data, revised_prompt=revised_prompt or _revised_prompt(value))


def _revised_prompt(value: Mapping[str, object]) -> str | None:
    for key in ("revised_prompt", "code"):
        raw_value = value.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()
    return None


class ImageService:
    """Own the OpenAI Images to ChatGLM image-generation vertical slice."""

    def __init__(self, client: GLMWebClient) -> None:
        self.client = client

    def generate(
        self,
        request: ImageGenerationRequest,
    ) -> ImageGenerationResult:
        client = self.client
        lease = client.request_queue.acquire(f"image:{request.model}")
        try:
            response, assistant_id = self._open_stream(
                request,
                preferred_account_index=client.get_preferred_account_index(lease.ticket),
            )
        except Exception:
            lease.release()
            raise

        accumulator = GLMImageEventAccumulator()
        try:
            if isinstance(response, dict):
                client.raise_for_event_error(response, stream=False)
                accumulator.consume_event(response)
                return self._require_result(accumulator.build_result(), response)

            for event in client.iter_sse_events(response):
                if not event:
                    continue
                client.raise_for_event_error(event, stream=False)
                status = accumulator.consume_event(event)
                if status in IMAGE_TERMINAL_STATUSES:
                    return self._require_result(accumulator.build_result(), event)

            return self._require_result(accumulator.build_result(), {})
        finally:
            if hasattr(response, "close"):
                response.close() # type: ignore
            client.delete_conversation(accumulator.conversation_id, assistant_id=assistant_id)
            lease.release()

    def _open_stream(
        self,
        request: ImageGenerationRequest,
        preferred_account_index: int | None = None,
    ):
        client = self.client
        prompt = request.prompt.strip()
        if not prompt:
            raise UpstreamAPIError(status_code=400, message="图片生成请求缺少 prompt")

        user_model = request.model.strip()
        request_body = json.dumps(
            build_glm_image_payload(
                request,
                assistant_id=client.config.glm_image_assistant_id,
                selected_model=DEFAULT_CHAT_MODEL_NAME,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        client.logger.info(
            "转发绘图请求 model=%s assistant_id=%s size=%s n=%s",
            user_model,
            client.config.glm_image_assistant_id,
            request.size.strip().lower(),
            request.n,
        )
        debug_dump(client.logger, client.config.debug_dump_all, "内部图像请求", request)
        debug_dump(client.logger, client.config.debug_dump_all, "转发到 GLM 的 image 原始请求体", request_body)

        def send_request(account_index: int, access_token: str):
            request = client.build_signed_request(
                client.config.chat_stream_url,
                data=request_body,
                method="POST",
                access_token=access_token,
            )
            debug_dump(
                client.logger,
                client.config.debug_dump_all,
                f"转发到 GLM 的 image 请求头 account={account_index}",
                dict(request.header_items()),
            )
            response = client.open_upstream_request(request)
            return self._prepare_response(response)

        response = client.call_with_account_failover(
            f"image:{user_model}",
            send_request,
            preferred_account_index=preferred_account_index,
        )
        return response, client.config.glm_image_assistant_id

    def _prepare_response(self, response):
        """Keep JSON image responses as objects and stream responses readable."""
        client = self.client
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" in content_type:
            try:
                payload = client.auth.read_json_response(response)
            finally:
                response.close()

            status = effective_event_status(payload)
            numeric_failure = is_nonzero_status(status)
            if numeric_failure:
                raise UpstreamAPIError(
                    status_code=502,
                    message=client.build_error_message(200, payload),
                    payload=payload,
                )
            return payload

        return client.wrap_stream_response(response)

    def _require_result(
        self,
        result: ImageGenerationResult,
        payload: dict[str, object],
    ) -> ImageGenerationResult:
        client = self.client
        if result.images:
            client.logger.info("绘图完成 返回图片数=%s", len(result.images))
            return result
        raise UpstreamAPIError(
            status_code=502,
            message="GLM 绘图请求已完成，但未返回可用图片结果。",
            payload=payload,
        )

    def download_as_base64(self, image_url: str) -> str:
        """Download one generated image with bounded, HTTP-only reads."""
        client = self.client
        parsed = urllib.parse.urlparse(image_url)
        if parsed.scheme == "data":
            try:
                header, encoded = image_url.split(",", 1)
                if ";base64" not in header.lower() or len(encoded) > _MAX_DATA_URL_LENGTH:
                    raise ValueError("data URL 必须使用 base64 编码")
                image_bytes = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise UpstreamAPIError(status_code=502, message="生成图片 data URL 无效。") from exc
            if len(image_bytes) > IMAGE_DOWNLOAD_MAX_BYTES:
                raise UpstreamAPIError(status_code=502, message="生成图片超过允许的大小。")
            return base64.b64encode(image_bytes).decode("ascii")

        _validate_public_image_url(image_url)

        try:
            opener = urllib.request.build_opener(_PublicImageRedirectHandler())
            with opener.open(image_url, timeout=client.config.request_timeout) as response:
                geturl = getattr(response, "geturl", None)
                final_url = geturl() if callable(geturl) else None
                if isinstance(final_url, str):
                    _validate_public_image_url(final_url)
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > IMAGE_DOWNLOAD_MAX_BYTES:
                            raise UpstreamAPIError(status_code=502, message="生成图片超过允许的大小。")
                    except ValueError:
                        pass

                image_bytes = bytearray()
                while True:
                    remaining = IMAGE_DOWNLOAD_MAX_BYTES - len(image_bytes) + 1
                    chunk = response.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    image_bytes.extend(chunk)
                    if len(image_bytes) > IMAGE_DOWNLOAD_MAX_BYTES:
                        raise UpstreamAPIError(status_code=502, message="生成图片超过允许的大小。")

                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type and not (
                    content_type.startswith("image/")
                    or content_type == "application/octet-stream"
                ):
                    raise UpstreamAPIError(status_code=502, message="生成图片响应类型无效。")
            return base64.b64encode(image_bytes).decode("ascii")
        except UpstreamAPIError:
            raise
        except Exception as exc:
            raise UpstreamAPIError(status_code=502, message=f"下载图片失败: {image_url} error={exc}") from exc


__all__ = [
    "GeneratedImage",
    "GLMImageEventAccumulator",
    "IMAGE_DOWNLOAD_MAX_BYTES",
    "IMAGE_SIZE_TO_ASPECT_RATIO",
    "IMAGE_TERMINAL_STATUSES",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageResponseConversionError",
    "ImageService",
    "MAX_IMAGE_COUNT",
    "SUPPORTED_IMAGE_RESPONSE_FORMATS",
    "build_glm_image_payload",
    "internal_to_openai_images_response",
    "openai_images_to_internal",
    "resolve_image_aspect_ratio",
]
