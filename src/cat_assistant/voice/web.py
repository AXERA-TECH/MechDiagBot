"""调试 UI 网页录音所需的语音服务：整段转写与整段合成。

网页按键录音不走本地唤醒/VAD；浏览器把 16 kHz 单声道 s16le 直接上传，
服务端转写后复用现有 ``/api/turn``，再把回复合成为 WAV 返回浏览器播放。
"""

from __future__ import annotations

from cat_assistant.voice.asr import QwenAsrClient
from cat_assistant.voice.config import VoiceConfig
from cat_assistant.voice.pcm import resample_s16, wav_bytes
from cat_assistant.voice.tts import VoxCpmClient, clean_for_speech

WEB_SAMPLE_RATE = 16_000


class WebVoiceService:
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.asr = QwenAsrClient(config.asr)
        self.tts = VoxCpmClient(
            config.tts, output_sample_rate=config.audio.output_sample_rate
        )

    async def transcribe(self, pcm: bytes, sample_rate: int = WEB_SAMPLE_RATE) -> str:
        if not pcm:
            return ""
        if sample_rate <= 0:
            raise ValueError("sample_rate 必须为正数")
        if sample_rate != WEB_SAMPLE_RATE:
            pcm = resample_s16(pcm, sample_rate, WEB_SAMPLE_RATE)
        return await self.asr.transcribe(pcm, WEB_SAMPLE_RATE)

    async def speak_wav(self, text: str) -> bytes:
        spoken = clean_for_speech(text)
        if not spoken:
            return b""
        pcm = await self.tts.synthesize(spoken)
        return wav_bytes(pcm, self.config.audio.output_sample_rate) if pcm else b""

    async def stream_pcm(self, text: str):
        """逐块产出播放采样率的单声道 s16le PCM；首块不再等整段合成。"""

        spoken = clean_for_speech(text)
        if not spoken:
            return
        async for chunk in self.tts.stream(spoken):
            if chunk:
                yield chunk

    async def close(self) -> None:
        await self.asr.close()
        await self.tts.close()


def voice_available() -> tuple[bool, str]:
    """检查网页语音所需的 Python 依赖是否可用。"""

    try:
        import numpy  # noqa: F401
    except ImportError:
        return False, "未安装语音依赖：pip install 'equip-ai-agent[voice]'"
    return True, ""


def voice_status(config: VoiceConfig) -> dict:
    available, error = voice_available()
    status = {
        "available": available,
        "asr": config.asr.base_url,
        "tts": config.tts.base_url,
        "wake": config.wake.provider if config.wake.enabled else "none",
    }
    if error:
        status["error"] = error
    return status


__all__ = [
    "WEB_SAMPLE_RATE",
    "WebVoiceService",
    "voice_available",
    "voice_status",
]
