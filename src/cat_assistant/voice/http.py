"""语音 HTTP 客户端公共工具（标准库实现，不引入额外依赖）。"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlsplit, urlunsplit


class VoiceServiceError(RuntimeError):
    """ASR/TTS 服务不可用或返回异常。"""


def validate_base_url(value: str, label: str = "base_url") -> str:
    text = (value or "").strip().rstrip("/")
    if not text:
        raise ValueError(f"{label} 不能为空")
    if not (text.startswith("http://") or text.startswith("https://")):
        raise ValueError(f"{label} 必须以 http:// 或 https:// 开头：{value}")
    return text


def endpoint_for(base_url: str, path: str) -> str:
    """把根地址、``/v1`` 地址或完整接口地址统一成目标端点。"""

    base = validate_base_url(base_url)
    if base.endswith(path):
        return base
    if path.startswith("/v1/") and base.endswith("/v1"):
        return base + path[3:]
    return base + path


def health_endpoint(base_url: str) -> str:
    parts = urlsplit(validate_base_url(base_url))
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def encode_multipart(
    fields: dict[str, str],
    *,
    file_field: str,
    filename: str,
    file_bytes: bytes,
    file_content_type: str = "application/octet-stream",
) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    chunks.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\nContent-Type: {file_content_type}\r\n\r\n'
        ).encode("utf-8")
    )
    chunks.append(file_bytes)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _open_sync(
    url: str,
    *,
    data: bytes | None,
    headers: dict[str, str] | None,
    timeout: float,
):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace").strip()
        except Exception:
            detail = ""
        raise VoiceServiceError(
            f"HTTP {exc.code}：{_error_text(detail) or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise VoiceServiceError(f"服务不可用：{exc.reason}") from exc
    except TimeoutError as exc:
        raise VoiceServiceError("请求超时") from exc


async def open_request(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
):
    """在线程中发请求，返回 urllib 响应对象（调用方负责 close）。"""

    return await asyncio.to_thread(
        _open_sync, url, data=data, headers=headers, timeout=timeout
    )


async def read_all(response, *, limit: int = 32_000_000) -> bytes:
    data = await asyncio.to_thread(response.read, limit + 1)
    if len(data) > limit:
        raise VoiceServiceError("服务返回内容过大")
    return data


async def read_chunk(response, size: int = 8192) -> bytes:
    # read1 一有数据就返回，避免为凑满 size 阻塞，降低流式首块延迟。
    reader = getattr(response, "read1", None)
    if callable(reader):
        return await asyncio.to_thread(reader, size)
    return await asyncio.to_thread(response.read, size)


def content_type(response) -> str:
    return (
        (response.headers.get("content-type", "") or "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )


def header_int(response, name: str) -> int:
    raw = response.headers.get(name)
    if raw is None:
        return 0
    try:
        return int(str(raw).strip())
    except ValueError:
        return 0


def json_error(body: bytes) -> str:
    """从错误响应体里提取 ``error.message`` 或文本，最长保留 300 字。"""

    text = body.decode("utf-8", "replace").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:300]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        if isinstance(error, str):
            return error[:300]
        if payload.get("message"):
            return str(payload["message"])[:300]
    return text[:300]


def _error_text(detail: str) -> str:
    return json_error(detail.encode("utf-8")) if detail else ""


async def warmup(base_url: str, *, timeout: float = 3.0) -> None:
    """访问 /health 预热连接；失败只记调试日志，不阻断启动。"""

    url = health_endpoint(base_url)
    try:
        response = await open_request(url, timeout=timeout)
        try:
            await read_all(response, limit=1_000_000)
        finally:
            response.close()
    except Exception:
        return


__all__ = [
    "VoiceServiceError",
    "content_type",
    "encode_multipart",
    "endpoint_for",
    "header_int",
    "health_endpoint",
    "json_error",
    "open_request",
    "read_all",
    "read_chunk",
    "validate_base_url",
    "warmup",
]
