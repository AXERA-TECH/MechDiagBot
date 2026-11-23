"""能量 VAD 与语音分段。

精简版只保留无额外依赖的 ``EnergyVad``：门限 = max(噪声底, noise_ratio ×
环境噪声估计)，底噪由非语音帧指数平滑得到，嘈杂环境自动抬高门限。分段器
支持起始前滚、短句能量收尾提前切句、单句最长时长和最短有效语音过滤。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from cat_assistant.voice.pcm import (
    AudioFrame,
    VoiceUtterance,
    pcm_duration_ms,
    rms,
)


@dataclass
class EnergyVad:
    """基于 RMS 的自适应能量 VAD，无额外依赖。"""

    floor: float = 0.0025
    noise_ratio: float = 2.0
    _noise: float = 0.0
    _attack: float = 0.05
    _decay: float = 0.10

    @property
    def gate(self) -> float:
        return max(max(0.0, self.floor), self._noise * max(1.0, self.noise_ratio))

    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        del sample_rate
        level = rms(pcm)
        if level < self.gate:
            alpha = self._attack if level > self._noise else self._decay
            self._noise = alpha * level + (1.0 - alpha) * self._noise
            return False
        return True

    def reset(self) -> None:
        # 每轮结束后归零重估，避免被播报残留回声抬高门限。
        self._noise = 0.0


@dataclass
class Segmenter:
    """把连续音频帧切成一句话。"""

    detector: EnergyVad
    sample_rate: int = 16_000
    frame_ms: int = 20
    silence_ms: int = 300
    min_speech_ms: int = 300
    max_utterance_s: int = 15
    pre_roll_ms: int = 300
    start_frames: int = 2
    early_release: bool = True
    early_release_silence_ms: int = 150
    early_release_max_speech_ms: int = 2000
    early_release_tail_ratio: float = 0.35

    def __post_init__(self) -> None:
        self.max_utterance_ms = max(1, int(self.max_utterance_s * 1000))
        self.max_pending_ms = self.pre_roll_ms + self.start_frames * self.frame_ms
        self._pending: deque[tuple[bytes, int, int]] = deque()
        self._pending_ms = 0
        self._active = False
        self._buffer = bytearray()
        self._start_ms = 0
        self._last_ms = 0
        self._voiced_ms = 0
        self._silence_ms = 0
        self._speech_run = 0
        self._speech_run_ms = 0
        self._peak_rms = 0.0
        self._tail_levels: deque[float] = deque(maxlen=3)
        self._last_voiced_at = 0.0

    @property
    def active(self) -> bool:
        return self._active

    def push(self, frame: AudioFrame) -> VoiceUtterance | None:
        pcm = frame.pcm
        if not pcm:
            return None
        duration = pcm_duration_ms(pcm, frame.sample_rate)
        speech = self.detector.is_speech(pcm, frame.sample_rate)

        if not self._active:
            self._pending.append((pcm, frame.timestamp_ms, duration))
            self._pending_ms += duration
            self._trim_pending()
            if speech:
                self._speech_run += 1
                self._speech_run_ms += duration
                if self._speech_run >= self.start_frames:
                    self._activate(frame, rms(pcm))
            else:
                self._speech_run = 0
                self._speech_run_ms = 0
            return None

        self._buffer.extend(pcm)
        self._last_ms = frame.timestamp_ms + duration
        if speech:
            self._silence_ms = 0
            self._voiced_ms += duration
            self._speech_run_ms += duration
            self._track_level(rms(pcm))
            self._last_voiced_at = time.perf_counter()
        else:
            self._silence_ms += duration
            self._speech_run_ms = 0

        if not speech and self._silence_ms >= self.silence_ms:
            return self._finish()
        if not speech and self._early_release_ready():
            return self._finish()
        if self._buffer_ms() >= self.max_utterance_ms:
            return self._finish()
        return None

    def flush(self) -> VoiceUtterance | None:
        """输入结束时把正在积累的语音切出来。"""

        if self._active:
            return self._finish()
        self._reset()
        return None

    def reset(self) -> None:
        self._reset()
        self.detector.reset()

    def _activate(self, frame: AudioFrame, level: float) -> None:
        self._active = True
        self._buffer = bytearray()
        self._start_ms = self._pending[0][1] if self._pending else frame.timestamp_ms
        for pcm, _, _ in self._pending:
            self._buffer.extend(pcm)
        self._pending.clear()
        self._pending_ms = 0
        self._voiced_ms = self._speech_run_ms
        self._silence_ms = 0
        self._last_ms = frame.timestamp_ms + pcm_duration_ms(frame.pcm, frame.sample_rate)
        self._peak_rms = level
        self._tail_levels.clear()
        self._last_voiced_at = time.perf_counter()
        if level > 0.0:
            self._tail_levels.append(level)

    def _track_level(self, level: float) -> None:
        if level > self._peak_rms:
            self._peak_rms = level
        self._tail_levels.append(level)

    def _early_release_ready(self) -> bool:
        if not self.early_release:
            return False
        if self._silence_ms < self.early_release_silence_ms:
            return False
        if self._voiced_ms > self.early_release_max_speech_ms:
            return False
        if self._peak_rms <= 0.0 or not self._tail_levels:
            return False
        tail = sum(self._tail_levels) / len(self._tail_levels)
        threshold = max(
            self._peak_rms * self.early_release_tail_ratio,
            float(self.detector.gate) * 1.5,
        )
        return tail <= threshold

    def _finish(self) -> VoiceUtterance | None:
        pcm = bytes(self._buffer)
        voiced_ms = self._voiced_ms
        start_ms, end_ms = self._start_ms, self._last_ms
        speech_end_at = self._last_voiced_at
        self._reset()
        if voiced_ms < self.min_speech_ms or not pcm:
            return None
        return VoiceUtterance(
            pcm=pcm,
            sample_rate=self.sample_rate,
            start_ms=start_ms,
            end_ms=end_ms,
            speech_end_at=speech_end_at,
        )

    def _reset(self) -> None:
        self._pending.clear()
        self._pending_ms = 0
        self._active = False
        self._buffer = bytearray()
        self._start_ms = 0
        self._last_ms = 0
        self._voiced_ms = 0
        self._silence_ms = 0
        self._speech_run = 0
        self._speech_run_ms = 0
        self._peak_rms = 0.0
        self._tail_levels.clear()
        self._last_voiced_at = 0.0

    def _trim_pending(self) -> None:
        while len(self._pending) > 1 and self._pending_ms > self.max_pending_ms:
            _, _, duration = self._pending.popleft()
            self._pending_ms -= duration

    def _buffer_ms(self) -> int:
        if not self._buffer:
            return 0
        return int(len(self._buffer) / 2 / self.sample_rate * 1000)


def build_segmenter(
    *,
    sample_rate: int = 16_000,
    frame_ms: int = 20,
    silence_ms: int = 300,
    min_speech_ms: int = 300,
    max_utterance_s: int = 15,
    pre_roll_ms: int = 300,
    energy_threshold: float = 0.0025,
    noise_ratio: float = 2.0,
) -> Segmenter:
    return Segmenter(
        EnergyVad(floor=energy_threshold, noise_ratio=noise_ratio),
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        silence_ms=silence_ms,
        min_speech_ms=min_speech_ms,
        max_utterance_s=max_utterance_s,
        pre_roll_ms=pre_roll_ms,
    )


__all__ = ["EnergyVad", "Segmenter", "build_segmenter"]
