"""VoxCPM2 TTS 客户端与流式分句。

服务协议为 OpenAI 兼容 ``/v1/audio/speech``：``stream=true`` 时响应体是
连续裸 s16le（VoxCPM2 默认 48 kHz），非流式则按 WAV 解码。声音克隆通过
``ref_audio``（base64 WAV）和可选 ``ref_text`` 传入。客户端把服务端采样率
重采样到播放采样率后逐块产出，供播放器边合成边播。
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from cat_assistant.voice.config import TtsSettings, resolve_path
from cat_assistant.voice.http import (
    VoiceServiceError,
    content_type,
    endpoint_for,
    header_int,
    json_error,
    open_request,
    read_all,
    read_chunk,
    warmup,
)
from cat_assistant.voice.pcm import StreamResampler, decode_wav_bytes

logger = logging.getLogger(__name__)

_PCM_CONTENT_TYPES = {
    "",
    "application/octet-stream",
    "binary/octet-stream",
    "audio/pcm",
    "audio/raw",
    "audio/x-pcm",
    "audio/s16le",
    "audio/l16",
}

_ENDINGS = set("。！？!?；;…\n")
_SOFT = set("，,、：: ")
_OPEN = {
    "“": "”",
    "‘": "’",
    "（": "）",
    "(": ")",
    "【": "】",
    "[": "]",
    "《": "》",
    "<": ">",
    "「": "」",
    "『": "』",
}
_CLOSERS = set(_OPEN.values())


@dataclass
class SentenceSplitter:
    """增量分句；引号、括号内部不切分，超长句在逗号处兜底。"""

    max_chars: int = 60
    _buffer: str = field(default="", init=False)
    _stack: list[str] = field(default_factory=list, init=False)

    def push(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        sentences: list[str] = []
        while True:
            cut = self._sentence_end(self._buffer)
            if cut is None:
                break
            if cut > self.max_chars:
                cut = self._long_cut(self._buffer) or self.max_chars
            sentences.append(self._buffer[:cut])
            self._buffer = self._buffer[cut:]
        while len(self._buffer) > self.max_chars:
            cut = self._long_cut(self._buffer) or self.max_chars
            sentences.append(self._buffer[:cut])
            self._buffer = self._buffer[cut:]
        return [item for item in (clean for clean in map(_clean, sentences)) if item]

    def flush(self) -> list[str]:
        text, self._buffer = self._buffer, ""
        self._stack.clear()
        cleaned = _clean(text)
        return [cleaned] if cleaned else []

    def reset(self) -> None:
        self._buffer = ""
        self._stack.clear()

    def split(self, text: str) -> list[str]:
        """一次性切分整段文本。"""

        return [*self.push(text), *self.flush()]

    def _sentence_end(self, text: str) -> int | None:
        self._stack = []
        index = 0
        while index < len(text):
            char = text[index]
            if self._track(char):
                index += 1
                continue
            if not self._stack and char in _ENDINGS:
                end = index + 1
                while end < len(text) and text[end] in _ENDINGS:
                    end += 1
                while end < len(text) and text[end] in _CLOSERS:
                    end += 1
                return end
            index += 1
        return None

    def _long_cut(self, text: str) -> int:
        limit = min(self.max_chars, len(text) - 1)
        self._stack = []
        last_soft = -1
        depth = 0
        for index, char in enumerate(text[: limit + 1]):
            if char in _OPEN:
                depth += 1
            elif char in _CLOSERS and depth:
                depth -= 1
            if depth == 0 and char in _SOFT:
                last_soft = index + 1
        if last_soft > 0:
            return last_soft
        if depth == 0:
            return limit
        return 0

    def _track(self, char: str) -> bool:
        if char in _OPEN:
            self._stack.append(_OPEN[char])
            return True
        if char == '"':
            if self._stack and self._stack[-1] == '"':
                self._stack.pop()
            else:
                self._stack.append('"')
            return True
        if self._stack and char == self._stack[-1]:
            self._stack.pop()
            return True
        return False


def _clean(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if not any(char.isalnum() for char in text):
        return ""
    return text


def clean_for_speech(text: str) -> str:
    """去掉 Markdown 标记、链接和多余空白，避免 TTS 念出符号。"""

    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text or "")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#>~|]+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


class VoxCpmClient:
    name = "voxcpm2"

    def __init__(self, settings: TtsSettings, *, output_sample_rate: int = 24_000):
        self.settings = settings
        self.base_url = settings.base_url
        self.output_sample_rate = int(output_sample_rate)
        self.source_sample_rate = int(settings.source_sample_rate)
        self.timeout_seconds = max(1.0, float(settings.timeout_seconds))
        if self.output_sample_rate <= 0 or self.source_sample_rate <= 0:
            raise ValueError("TTS 采样率必须为正数")
        self.ref_audio = _ref_audio_b64(settings.ref_audio)
        self.ref_text = (settings.ref_text or "").strip()
        self.extra = {
            key: _numeric(value)
            for key, value in {
                "seed": settings.seed,
                "max_steps": settings.max_steps,
                "inference_timesteps": settings.inference_timesteps,
                "cfg_value": settings.cfg_value,
                "temperature": settings.temperature,
            }.items()
            if value not in (None, "")
        }

    @property
    def endpoint(self) -> str:
        return endpoint_for(self.base_url, "/v1/audio/speech")

    async def warmup(self) -> None:
        await warmup(self.base_url)

    def _payload(self, text: str, *, stream: bool) -> dict:
        payload: dict = {
            "input": text,
            # 流式只支持裸 PCM；非流式按 WAV 返回，最稳妥。
            "response_format": "pcm" if stream else "wav",
        }
        if self.settings.model:
            payload["model"] = self.settings.model
        if self.settings.voice:
            payload["voice"] = self.settings.voice
        if self.ref_audio:
            payload["ref_audio"] = self.ref_audio
        if self.ref_text:
            payload["ref_text"] = self.ref_text
        if stream:
            payload["stream"] = True
        payload.update(self.extra)
        return payload

    async def synthesize(self, text: str) -> bytes:
        """合成整段文本，返回播放采样率的单声道 s16le。"""

        text = clean_for_speech(text)
        if not text:
            return b""
        if not self.settings.streaming:
            return await self._synthesize_once(text)
        chunks = [chunk async for chunk in self.stream(text)]
        return b"".join(chunks)

    async def _synthesize_once(self, text: str) -> bytes:
        body = json.dumps(self._payload(text, stream=False), ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        response = await open_request(
            self.endpoint, data=body, headers=headers, timeout=self.timeout_seconds
        )
        try:
            raw = await read_all(response)
            ctype = content_type(response)
        finally:
            response.close()
        if not raw:
            raise VoiceServiceError("TTS 返回空音频")
        if "json" in ctype:
            raise VoiceServiceError(f"TTS 返回错误：{json_error(raw)}")
        pcm, _ = decode_wav_bytes(raw, self.output_sample_rate)
        if not pcm:
            raise VoiceServiceError("TTS 音频解码后为空")
        return pcm

    async def stream(self, text: str):
        """异步产出播放采样率的 PCM 块；服务不支持流式时退化为整段。"""

        text = clean_for_speech(text)
        if not text:
            return
        if not self.settings.streaming:
            pcm = await self._synthesize_once(text)
            if pcm:
                yield pcm
            return

        body = json.dumps(self._payload(text, stream=True), ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        response = await open_request(
            self.endpoint, data=body, headers=headers, timeout=self.timeout_seconds
        )
        try:
            ctype = content_type(response)
            if "json" in ctype:
                raw = await read_all(response)
                raise VoiceServiceError(f"TTS 返回错误：{json_error(raw)}")
            if ctype not in _PCM_CONTENT_TYPES:
                # 服务忽略 stream 返回整段 WAV：收完再解码，退化为单块。
                raw = await read_all(response)
                if not raw:
                    raise VoiceServiceError("TTS 流式响应没有音频")
                try:
                    pcm, _ = decode_wav_bytes(raw, self.output_sample_rate)
                except ValueError as exc:
                    raise VoiceServiceError(str(exc)) from exc
                if pcm:
                    yield pcm
                return

            source_rate = header_int(response, "X-Sample-Rate") or self.source_sample_rate
            if source_rate <= 0:
                raise VoiceServiceError("TTS X-Sample-Rate 必须为正数")
            resampler = StreamResampler(source_rate, self.output_sample_rate)
            received = 0
            while True:
                try:
                    data = await read_chunk(response)
                except VoiceServiceError:
                    raise
                except Exception as exc:
                    raise VoiceServiceError(
                        f"TTS 流式读取失败：{type(exc).__name__}: {exc}"
                    ) from exc
                if not data:
                    break
                pcm = resampler.push(data)
                if pcm:
                    received += len(pcm)
                    yield pcm
            tail = resampler.flush()
            if tail:
                received += len(tail)
                yield tail
            if not received:
                raise VoiceServiceError("TTS 流式响应没有音频")
        finally:
            response.close()

    async def close(self) -> None:
        return


def _ref_audio_b64(value: str) -> str:
    """把 ref_audio 配置转成服务端要的裸 base64 WAV；支持路径或 data URL。"""

    text = (value or "").strip()
    if not text:
        return ""
    if text.startswith("data:"):
        return text.partition(",")[2].strip()
    if text.startswith("base64:"):
        return text[len("base64:") :]
    path = resolve_path(text)
    if not path.is_file():
        raise ValueError(f"TTS ref_audio 文件不存在：{text}")
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"TTS ref_audio 不是 WAV 文件：{text}")
    return base64.b64encode(data).decode("ascii")


def _numeric(value: str):
    text = str(value).strip()
    if not text:
        return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


__all__ = [
    "SentenceSplitter",
    "VoxCpmClient",
    "clean_for_speech",
]
