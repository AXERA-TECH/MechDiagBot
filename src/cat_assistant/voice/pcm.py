"""单声道 s16le PCM 工具：分帧、重采样、WAV 读写、能量与提示音。"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_WIDTH = 2  # s16le


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """一帧等间隔 PCM；pcm 为单声道 s16le。"""

    pcm: bytes
    sample_rate: int = 16_000
    timestamp_ms: int = 0


@dataclass(frozen=True, slots=True)
class VoiceUtterance:
    """VAD 切出的一段用户语音。"""

    pcm: bytes
    sample_rate: int
    start_ms: int = 0
    end_ms: int = 0
    speech_end_at: float = 0.0

    @property
    def duration_ms(self) -> int:
        return pcm_duration_ms(self.pcm, self.sample_rate)


def frame_bytes(sample_rate: int, frame_ms: int) -> int:
    return int(sample_rate * frame_ms / 1000) * SAMPLE_WIDTH


def pcm_duration_ms(pcm: bytes, sample_rate: int) -> int:
    if not pcm or sample_rate <= 0:
        return 0
    return int(len(pcm) / SAMPLE_WIDTH / sample_rate * 1000)


def silence(ms: int, sample_rate: int) -> bytes:
    return b"\x00\x00" * int(sample_rate * ms / 1000)


def iter_pcm_frames(
    pcm: bytes,
    sample_rate: int,
    frame_ms: int = 20,
    *,
    pad: bool = True,
):
    """按固定时长切帧；最后一帧补零，保证 VAD 拿到完整帧。"""

    size = frame_bytes(sample_rate, frame_ms)
    if size <= 0:
        raise ValueError("frame_ms 必须为正数")
    for index in range(0, len(pcm), size):
        chunk = pcm[index : index + size]
        if len(chunk) < size and pad:
            chunk += b"\x00" * (size - len(chunk))
        if chunk:
            yield chunk


def resample_s16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """s16le 重采样；整数倍降采样用等间隔抽取，其余线性插值。"""

    if not pcm or source_rate == target_rate:
        return pcm
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("采样率必须为正数")
    samples = np.frombuffer(pcm, dtype="<i2")
    if samples.size == 0:
        return b""
    if source_rate > target_rate and source_rate % target_rate == 0:
        return samples[:: source_rate // target_rate].tobytes()
    count = max(1, int(round(samples.size / source_rate * target_rate)))
    target_positions = np.linspace(0, samples.size - 1, num=count, dtype=np.float64)
    values = np.interp(
        target_positions,
        np.arange(samples.size, dtype=np.float64),
        samples.astype(np.float32),
    )
    return np.clip(np.round(values), -32768, 32767).astype("<i2").tobytes()


class StreamResampler:
    """流式分块重采样，保证块边界上的采样相位连续。

    整数倍降采样（VoxCPM2 48k → 24k/16k）把不足一个输出采样的尾部和
    奇数尾字节留到下一块；非整数倍率按块线性插值（原型质量）。
    """

    def __init__(self, source_rate: int, target_rate: int):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("采样率必须为正数")
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self.factor = (
            self.source_rate // self.target_rate
            if self.source_rate > self.target_rate
            and self.source_rate % self.target_rate == 0
            else 1
        )
        self._carry = b""

    def push(self, data: bytes) -> bytes:
        if not data:
            return b""
        buffered = self._carry + data
        # HTTP 分块边界不保证按采样对齐，奇数尾字节留到下一块。
        hold = len(buffered) % 2
        if self.factor > 1:
            hold += ((len(buffered) - hold) // 2 % self.factor) * 2
        if hold:
            self._carry, data = buffered[-hold:], buffered[:-hold]
        else:
            self._carry, data = b"", buffered
        return resample_s16(data, self.source_rate, self.target_rate) if data else b""

    def flush(self) -> bytes:
        tail = self._carry[: len(self._carry) - len(self._carry) % 2]
        self._carry = b""
        return resample_s16(tail, self.source_rate, self.target_rate) if tail else b""


def s16le_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_s16le(values: np.ndarray) -> bytes:
    return np.clip(np.round(values * 32767), -32768, 32767).astype("<i2").tobytes()


def rms(pcm: bytes) -> float:
    """返回 0..1 的均方根能量。"""

    if not pcm:
        return 0.0
    values = s16le_to_float32(pcm)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def fade_in_out(values: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size < 2 or sample_rate <= 0 or seconds <= 0:
        return values
    fade = min(values.size // 2, max(1, int(sample_rate * float(seconds))))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    values[:fade] *= ramp
    values[-fade:] *= ramp[::-1]
    return values


def tone(seconds: float, sample_rate: int, *, hz: float = 880.0, volume: float = 0.25) -> bytes:
    """生成带淡入淡出的提示音。"""

    count = max(1, int(sample_rate * max(0.01, seconds)))
    positions = np.arange(count, dtype=np.float32) / sample_rate
    values = min(0.9, max(0.0, volume)) * np.sin(2 * np.pi * hz * positions)
    return float32_to_s16le(fade_in_out(values, sample_rate, 0.01))


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """把单声道 s16le PCM 封装成内存中的 WAV。"""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def decode_wav_bytes(data: bytes, target_rate: int) -> tuple[bytes, int]:
    """解码 WAV 为单声道 s16le；16bit PCM 走标准库，其余格式退回 soundfile。"""

    if not data:
        return b"", int(target_rate)
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        if width != SAMPLE_WIDTH:
            raise ValueError("仅支持 16bit WAV")
        if channels > 1:
            samples = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
            frames = np.round(samples.mean(axis=1)).astype("<i2").tobytes()
        return resample_s16(frames, int(rate), target_rate), int(target_rate)
    except (wave.Error, EOFError, ValueError):
        pass

    try:
        import soundfile as sf

        values, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except ImportError as exc:
        raise ValueError(
            "TTS 返回的音频不是 16bit PCM WAV；安装 soundfile 后可解码其他格式"
        ) from exc
    except Exception as exc:
        raise ValueError(f"无法解码 TTS 音频：{type(exc).__name__}: {exc}") from exc
    if not values.size:
        return b"", int(target_rate)
    pcm = float32_to_s16le(values.mean(axis=1))
    return resample_s16(pcm, int(rate), target_rate), int(target_rate)


def read_wav(path: str | Path, *, target_rate: int | None = None) -> tuple[bytes, int]:
    """读取 WAV 为单声道 s16le；其他容器格式在安装 soundfile 后支持。"""

    data = Path(path).read_bytes()
    if target_rate is not None:
        return decode_wav_bytes(data, target_rate)
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = int(handle.getframerate())
            frames = handle.readframes(handle.getnframes())
        if width != SAMPLE_WIDTH:
            raise ValueError("仅支持 16bit WAV")
        if channels > 1:
            samples = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
            frames = np.round(samples.mean(axis=1)).astype("<i2").tobytes()
        return frames, rate
    except (wave.Error, EOFError, ValueError):
        pass

    try:
        import soundfile as sf

        values, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except ImportError as exc:
        raise ValueError("仅支持 16bit PCM WAV；安装 soundfile 后可读取其他格式") from exc
    if not values.size:
        return b"", int(rate)
    return float32_to_s16le(values.mean(axis=1)), int(rate)


def write_wav(path: str | Path, pcm: bytes, sample_rate: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


__all__ = [
    "AudioFrame",
    "SAMPLE_WIDTH",
    "StreamResampler",
    "VoiceUtterance",
    "decode_wav_bytes",
    "fade_in_out",
    "float32_to_s16le",
    "frame_bytes",
    "iter_pcm_frames",
    "pcm_duration_ms",
    "read_wav",
    "resample_s16",
    "rms",
    "s16le_to_float32",
    "silence",
    "tone",
    "wav_bytes",
    "write_wav",
]
