"""ChatGLM file and attachment upload helpers."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import mimetypes
import socket
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from email.generator import _make_boundary  # type: ignore
from typing import TYPE_CHECKING

from ..infrastructure.logging import debug_dump
from .errors import UpstreamAPIError

if TYPE_CHECKING:
    from .client import GLMWebClient


FILE_UPLOAD_URL_SUFFIX = "/backend-api/assistant/file_upload"
FILE_SIZE_LIMIT = 100 * 1024 * 1024
_MAX_DATA_URL_LENGTH = 4 * ((FILE_SIZE_LIMIT + 2) // 3)


def _validate_public_file_url(file_url: str) -> None:
    """Reject local/private destinations before downloading a reference file."""
    parsed = urllib.parse.urlparse(file_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("附件 URL 只支持不带凭据的 HTTP(S) 地址。")

    try:
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    except ValueError as exc:
        raise ValueError("附件 URL 主机名无效。") from exc
    if (
        not hostname
        or hostname in {"localhost", "localhost.localdomain"}
        or hostname.endswith(".local")
        or hostname.endswith(".localhost")
    ):
        raise ValueError("附件 URL 指向本地地址。")

    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        literal_address = None
    if literal_address is not None:
        if not literal_address.is_global:
            raise ValueError("附件 URL 指向非公网地址。")
        return

    try:
        address_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("附件 URL 无法解析。") from exc
    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for info in address_infos:
        try:
            addresses.add(ipaddress.ip_address(info[4][0]))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("附件 URL 解析结果无效。") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("附件 URL 指向非公网地址。")


class _PublicFileRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate every redirect target before urllib follows it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_file_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _safe_filename(value: str, fallback: str) -> str:
    """Keep an untrusted URL or form filename safe for multipart headers."""
    filename = urllib.parse.unquote(value).replace("\\", "/").rsplit("/", 1)[-1]
    filename = "".join("_" if char in '\x00\r\n"' else char for char in filename).strip()
    return filename[:255] or fallback


def _build_uploaded_reference(
    result: object,
    file_url: str,
    is_image: bool,
) -> dict[str, object] | None:
    """Build a ChatGLM attachment reference from an upload response."""
    if not isinstance(result, dict):
        return None

    raw_file_id = result.get("file_id") or result.get("source_id")
    file_id = str(raw_file_id).strip() if raw_file_id is not None else ""
    raw_result_url = result.get("file_url")
    result_url = str(raw_result_url).strip() if isinstance(raw_result_url, str) else ""

    if is_image:
        image_url = result_url or file_id
        if not image_url:
            return None
        return {"type": "image_url", "image_url": {"url": image_url}}

    if not file_id:
        return None
    return {
        "type": "file",
        "file": [{"source_id": file_id, "file_url": result_url or file_url}],
    }


@dataclass(slots=True)
class FileService:
    """Upload ChatGLM text-chat attachments and build provider references."""

    client: GLMWebClient

    def upload_referenced_files(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        """Upload file and image references found in GLM chat messages."""
        refs: list[dict[str, object]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "image_url":
                    url = item.get("image_url", {}).get("url")
                    if isinstance(url, str) and url:
                        refs.append(self._upload_file_reference(url, is_image=True))
                elif item_type == "file":
                    url = item.get("file_url", {}).get("url")
                    if isinstance(url, str) and url:
                        refs.append(self._upload_file_reference(url, is_image=False))
        if refs:
            self.client.logger.info("上传附件完成 成功数=%s", len(refs))
        return refs

    def _upload_file_reference(self, file_url: str, is_image: bool) -> dict[str, object]:
        try:
            filename, mime_type, payload = self.fetch_file_payload(file_url)
            boundary = _make_boundary()
            body = self._build_multipart(boundary, filename, mime_type, payload)
            upload_url = f"{self.client.config.glm_base_url}{FILE_UPLOAD_URL_SUFFIX}"
            debug_dump(
                self.client.logger,
                self.client.config.debug_dump_all,
                f"准备上传附件 url={file_url} filename={filename} mime={mime_type}",
                {"filename": filename, "mime_type": mime_type, "bytes": len(payload)},
            )

            def send_request(account_index: int, access_token: str):
                request = self.client.build_signed_request(
                    upload_url,
                    method="POST",
                    access_token=access_token,
                    data=body,
                    referer="https://chatglm.cn/",
                    extra_headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                )
                debug_dump(
                    self.client.logger,
                    self.client.config.debug_dump_all,
                    f"转发到 GLM 的 file_upload 请求头 account={account_index}",
                    dict(request.header_items()),
                )
                debug_dump(
                    self.client.logger,
                    self.client.config.debug_dump_all,
                    f"转发到 GLM 的 file_upload 原始请求体 account={account_index}",
                    body,
                )
                return self.client.open_upstream_request(request)

            with self.client.call_with_account_failover("file_upload", send_request) as response: # type: ignore
                result = self.client.auth.read_json_response(response).get("result", {})
            debug_dump(self.client.logger, self.client.config.debug_dump_all, "GLM 文件上传响应 result", result)
            reference = _build_uploaded_reference(result, file_url, is_image)
            if reference is None:
                raise UpstreamAPIError(
                    status_code=502,
                    message="GLM 文件上传响应缺少可用附件引用。",
                )
            return reference
        except (ValueError, UpstreamAPIError):
            raise
        except Exception as exc:
            self.client.logger.warning("上传附件失败 url=%s error=%s", file_url, exc)
            raise UpstreamAPIError(status_code=502, message="GLM 附件上传失败。") from exc

    def fetch_file_payload(self, file_url: str) -> tuple[str, str, bytes]:
        """Resolve a data or remote URL into an uploadable file payload."""
        if file_url.lower().startswith("data:"):
            try:
                header, encoded = file_url.split(",", 1)
            except ValueError as exc:
                raise ValueError("data URL 格式无效。") from exc
            if len(encoded) > _MAX_DATA_URL_LENGTH or not any(
                part.strip().lower() == "base64" for part in header.split(";")[1:]
            ):
                raise ValueError("data URL 必须是 100MB 以内的 base64 数据。")
            mime_type = header.split(";")[0][5:] or "application/octet-stream"
            extension = mimetypes.guess_extension(mime_type) or ".bin"
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("data URL 的 base64 数据无效。") from exc
            if len(payload) > FILE_SIZE_LIMIT:
                raise ValueError("文件超过 100MB，拒绝上传。")
            return f"upload-{uuid.uuid4().hex}{extension}", mime_type, payload

        _validate_public_file_url(file_url)
        parsed = urllib.parse.urlparse(file_url)
        filename = _safe_filename(
            parsed.path.rsplit("/", 1)[-1],
            f"upload-{uuid.uuid4().hex}.bin",
        )
        opener = urllib.request.build_opener(_PublicFileRedirectHandler())
        with opener.open(file_url, timeout=self.client.config.request_timeout) as response:
            payload = response.read(FILE_SIZE_LIMIT + 1)
            if len(payload) > FILE_SIZE_LIMIT:
                raise ValueError("文件超过 100MB，拒绝上传。")
            mime_type = response.headers.get_content_type()
        mime_type = mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return filename, mime_type, payload

    def _build_multipart(self, boundary: str, filename: str, mime_type: str, payload: bytes) -> bytes:
        start = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
        end = f"\r\n--{boundary}--\r\n".encode("utf-8")
        return start + payload + end


__all__ = [
    "FILE_SIZE_LIMIT",
    "FILE_UPLOAD_URL_SUFFIX",
    "FileService",
    "_build_uploaded_reference",
]
