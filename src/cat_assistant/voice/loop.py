"""实时语音交互循环：唤醒 → VAD → Qwen ASR → AgentLoop → VoxCPM2 播报。

精简版不做 AEC、声纹和语义轮次检测：播报期间麦克风输入会被丢弃，
播报结束后清空积压帧并重置 VAD，避免把自身播报回采成下一轮提问。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import replace

from cat_assistant.domain.models import AssistantResponse, Utterance
from cat_assistant.domain.ports import AgentHandlePort
from cat_assistant.voice.asr import QwenAsrClient
from cat_assistant.voice.audio import (
    AudioSink,
    AudioSource,
    MicSource,
    SpeakerSink,
)
from cat_assistant.voice.config import VoiceConfig, resolve_path
from cat_assistant.voice.pcm import (
    AudioFrame,
    VoiceUtterance,
    read_wav,
    tone,
)
from cat_assistant.voice.player import SpeechPlayer
from cat_assistant.voice.tts import VoxCpmClient
from cat_assistant.voice.vad import Segmenter, build_segmenter
from cat_assistant.voice.wake import NullWakeDetector, WakeDetector, build_wake

logger = logging.getLogger(__name__)


def new_session_id() -> str:
    return f"voice-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"


class VoiceLoop:
    def __init__(
        self,
        config: VoiceConfig,
        app: AgentHandlePort,
        *,
        asr: QwenAsrClient | None = None,
        tts: VoxCpmClient | None = None,
        source: AudioSource | None = None,
        sink: AudioSink | None = None,
        wake: WakeDetector | None = None,
        segmenter: Segmenter | None = None,
    ):
        self.config = config
        self.app = app
        self.asr = asr or QwenAsrClient(config.asr)
        self.tts = tts or VoxCpmClient(
            config.tts, output_sample_rate=config.audio.output_sample_rate
        )
        self.source = source or MicSource(
            device=config.audio.input_device,
            sample_rate=config.audio.sample_rate,
            frame_ms=config.audio.frame_ms,
        )
        self.sink = sink or SpeakerSink(
            device=config.audio.output_device,
            sample_rate=config.audio.output_sample_rate,
        )
        self.player = SpeechPlayer(
            self.tts, self.sink, output_sample_rate=config.audio.output_sample_rate
        )
        self.segmenter = segmenter or build_segmenter(
            sample_rate=config.audio.sample_rate,
            frame_ms=config.audio.frame_ms,
            silence_ms=config.vad.silence_ms,
            min_speech_ms=config.vad.min_speech_ms,
            max_utterance_s=config.vad.max_utterance_s,
            pre_roll_ms=config.vad.pre_roll_ms,
            energy_threshold=config.vad.energy_threshold,
            noise_ratio=config.vad.noise_ratio,
        )
        self.session_id = config.session.session_id or new_session_id()
        self._wake: WakeDetector | None = wake
        self._wake_enabled = bool(config.wake.enabled) or not isinstance(
            wake, (NullWakeDetector, type(None))
        )
        self._followup_until = 0.0
        self._ack_pcm = b""
        self._closed = False

    # ------------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        if self._wake is None:
            self._wake = build_wake(self.config.wake) if self.config.wake.enabled else NullWakeDetector()
        await self.asr.warmup()
        await self.tts.warmup()
        if self._wake_enabled and self.config.wake.enabled:
            self._ack_pcm = await self._prepare_ack()
        logger.info(
            "语音交互已就绪：ASR=%s TTS=%s 唤醒=%s 会话=%s",
            self.config.asr.base_url,
            self.config.tts.base_url,
            self.config.wake.provider if self._wake_enabled else "none",
            self.session_id,
        )

    async def run(self) -> None:
        try:
            await self.start()
            async for frame in self.source.frames():
                await self.on_frame(frame)
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.player.close()
        finally:
            try:
                await self.source.close()
            finally:
                close_wake = getattr(self._wake, "close", None)
                if callable(close_wake):
                    close_wake()
                await self.asr.close()
                await self.tts.close()
                shutdown = getattr(self.app, "shutdown", None)
                if callable(shutdown):
                    await shutdown()

    # ------------------------------------------------------------------ 实时帧

    async def on_frame(self, frame: AudioFrame) -> None:
        if self._wake_enabled and not self._followup_active():
            wake = self._wake
            if wake is not None:
                event = wake.detect(frame)
                if event.detected:
                    await self._on_wake(event.label)
                    return
            # 未唤醒时丢弃语音，避免把唤醒词或环境说话送进 ASR。
            self.segmenter.reset()
            return

        utterance = self.segmenter.push(frame)
        if utterance is not None:
            await self.handle_utterance(utterance)

    async def _on_wake(self, label: str) -> None:
        logger.info("检测到唤醒词：%s", label)
        self._open_followup()
        self.segmenter.reset()
        if self._ack_pcm:
            await self.player.play_pcm(self._ack_pcm)
        # 确认音可能被麦克风回采：播完再丢一次积压帧并重置 VAD。
        await self.source.discard_pending()
        self.segmenter.reset()

    async def handle_utterance(
        self, utterance: VoiceUtterance
    ) -> tuple[str, AssistantResponse | None]:
        started = time.perf_counter()
        asr_started = time.perf_counter()
        try:
            text = await self.asr.transcribe(utterance.pcm, utterance.sample_rate)
        except Exception as exc:
            logger.warning("ASR 识别失败：%s", exc)
            return "", None
        asr_ms = (time.perf_counter() - asr_started) * 1000
        if not text:
            logger.info("未识别到有效语音（%.0fms），忽略", asr_ms)
            return "", None
        logger.info("识别（%.0fms，语音 %.1fs）：%s", asr_ms, utterance.duration_ms / 1000, text)

        agent_started = time.perf_counter()
        try:
            response = await self.app.handle(
                Utterance(
                    text=text,
                    session_id=self.session_id,
                    machine_id=self.config.session.machine_id,
                    operator_id=self.config.session.operator_id or None,
                )
            )
        except Exception as exc:
            logger.warning("Agent 处理失败：%s", exc)
            return text, None
        agent_ms = (time.perf_counter() - agent_started) * 1000
        logger.info("回复（%.0fms）：%s", agent_ms, response.text)
        if response.text.strip():
            await self.player.say(response.text)
        # 播报结束后丢弃思考期间和播报回采的积压帧，避免自问自答。
        await self.source.discard_pending()
        self.segmenter.reset()
        self._open_followup()
        logger.info(
            "本轮完成：ASR %.0fms + Agent %.0fms + 播报，总计 %.1fs",
            asr_ms,
            agent_ms,
            time.perf_counter() - started,
        )
        return text, response

    # ------------------------------------------------------------------ 辅助

    def _open_followup(self) -> None:
        timeout = max(0, int(self.config.wake.followup_timeout_ms)) / 1000
        self._followup_until = time.monotonic() + timeout

    def _followup_active(self) -> bool:
        return time.monotonic() < self._followup_until

    async def _prepare_ack(self) -> bytes:
        mode = self.config.wake.ack.strip().lower()
        rate = self.config.audio.output_sample_rate
        if mode in {"", "none", "off"}:
            return b""
        if mode == "tone":
            return tone(0.12, rate, hz=880.0, volume=0.25)
        if mode == "voice":
            if self.config.wake.ack_audio:
                try:
                    pcm, _ = read_wav(resolve_path(self.config.wake.ack_audio), target_rate=rate)
                    return pcm
                except Exception as exc:
                    logger.warning("唤醒确认音读取失败，改用提示音：%s", exc)
                    return tone(0.12, rate)
            try:
                return await asyncio.wait_for(
                    self.tts.synthesize(self.config.wake.ack_text),
                    timeout=max(1.0, float(self.config.wake.ack_timeout_seconds)),
                )
            except Exception as exc:
                logger.warning("唤醒确认音预合成失败，改用提示音：%s", exc)
                return tone(0.12, rate)
        logger.warning("未知 wake.ack=%s，改用提示音", mode)
        return tone(0.12, rate)


def offline_config(config: VoiceConfig) -> VoiceConfig:
    """WAV 离线模式：关闭唤醒；输入按 16k 处理，输出保持播放采样率。"""

    return replace(
        config,
        audio=replace(
            config.audio,
            input_device="default",
            output_device="default",
        ),
        wake=replace(config.wake, provider="none"),
    )


def frame_for(pcm: bytes, sample_rate: int, timestamp_ms: int = 0) -> AudioFrame:
    return AudioFrame(pcm, sample_rate, timestamp_ms)


__all__ = [
    "VoiceLoop",
    "frame_for",
    "new_session_id",
    "offline_config",
]
