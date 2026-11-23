"""Qwen ASR 客户端：兼容 OpenAI ``/v1/audio/transcriptions``。

Axea GPU 服务（Qwen3-ASR）与 AX8860 板端 llama.cpp Qwen3-ASR 共用该契约：
multipart ``file`` + ``response_format=json`` + ``language=zh``，返回
``{"text", "language"}``。客户端把 16 kHz 单声道 s16le PCM 封成 WAV 后上传。
"""

from __future__ import annotations

import json
import logging

from cat_assistant.voice.config import AsrSettings
from cat_assistant.voice.http import (
    VoiceServiceError,
    encode_multipart,
    endpoint_for,
    json_error,
    open_request,
    read_all,
    warmup,
)
from cat_assistant.voice.pcm import wav_bytes

logger = logging.getLogger(__name__)


class QwenAsrClient:
    name = "qwen"

    def __init__(self, settings: AsrSettings):
        self.settings = settings
        self.base_url = settings.base_url
        self.timeout_seconds = max(1.0, float(settings.timeout_seconds))

    @property
    def endpoint(self) -> str:
        return endpoint_for(self.base_url, "/v1/audio/transcriptions")

    async def warmup(self) -> None:
        await warmup(self.base_url)

    async def transcribe(self, pcm: bytes, sample_rate: int = 16_000) -> str:
        """把一段单声道 s16le PCM 识别成文本；空音频返回空串。"""

        if not pcm:
            return ""
        if sample_rate <= 0 or len(pcm) % 2:
            raise ValueError("ASR 输入必须是有效采样率的 s16le PCM")

        fields = {"response_format": "json"}
        if self.settings.model:
            fields["model"] = self.settings.model
        if self.settings.language and self.settings.language != "auto":
            fields["language"] = self.settings.language
        body, content_type = encode_multipart(
            fields,
            file_field="file",
            filename="utterance.wav",
            file_bytes=wav_bytes(pcm, sample_rate),
            file_content_type="audio/wav",
        )
        headers = {"Content-Type": content_type}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        response = await open_request(
            self.endpoint,
            data=body,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        try:
            raw = await read_all(response)
        finally:
            response.close()
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise VoiceServiceError(
                f"ASR 返回不是 JSON：{raw[:200].decode('utf-8', 'replace')}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise VoiceServiceError(f"ASR 返回缺少 text：{json_error(raw)}")
        text = payload["text"].strip()
        logger.debug("Qwen ASR：%s", text)
        return text

    async def close(self) -> None:
        return


__all__ = ["QwenAsrClient"]
