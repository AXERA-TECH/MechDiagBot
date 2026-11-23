"""语音交互配置：JSON 文件 + 环境变量覆盖。

精简链路只保留本地唤醒、能量 VAD、Qwen ASR 与 VoxCPM2 TTS 所需的字段。
读取顺序（先找到的可读文件生效）：显式 ``path``、``$CAT_VOICE_CONFIG``、
``runtime/voice-config.json``、``config/voice-config.json``；都不存在时使用
内置默认值。环境变量优先于文件，便于沿用 Axea 语音模块的既有部署变量。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_WAKE_MODEL = "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
DEFAULT_WAKE_WORD = "你好，阿克西娅"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    text = _env(name)
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    text = _env(name)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    text = _env(name).lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    return value if isinstance(value, dict) else {}


def _text(raw: dict[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    return default if value is None else str(value).strip()


def _number(raw: dict[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(raw: dict[str, Any], key: str, default: int) -> int:
    return int(_number(raw, key, float(default)))


def _boolean(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _device(value: Any) -> str | int:
    if value is None or value == "" or str(value).strip().lower() == "default":
        return "default"
    text = str(value).strip()
    return int(text) if text.isdigit() else text


@dataclass(frozen=True, slots=True)
class AudioSettings:
    # 采集/唤醒/VAD/ASR 采样率；Qwen ASR 服务按 16 kHz 处理。
    input_device: str | int = "default"
    output_device: str | int = "default"
    sample_rate: int = 16_000
    output_sample_rate: int = 24_000
    frame_ms: int = 20


@dataclass(frozen=True, slots=True)
class VadSettings:
    silence_ms: int = 300
    min_speech_ms: int = 300
    max_utterance_s: int = 15
    pre_roll_ms: int = 300
    energy_threshold: float = 0.0025
    noise_ratio: float = 2.0


@dataclass(frozen=True, slots=True)
class WakeSettings:
    provider: str = "sherpa_onnx"  # sherpa_onnx | none
    model: str = DEFAULT_WAKE_MODEL
    word: str = DEFAULT_WAKE_WORD
    keywords_file: str = ""
    threshold: float = 0.25
    keywords_score: float = 1.0
    num_threads: int = 2
    ack: str = "voice"  # voice | tone | none
    ack_text: str = "我在"
    ack_audio: str = ""
    ack_timeout_seconds: float = 10.0
    followup_timeout_ms: int = 15_000

    @property
    def enabled(self) -> bool:
        return self.provider.strip().lower() not in {"", "none", "off", "disabled"}


@dataclass(frozen=True, slots=True)
class AsrSettings:
    base_url: str = "http://127.0.0.1:8021"
    api_key: str = ""
    model: str = ""
    language: str = "zh"
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class TtsSettings:
    base_url: str = "http://127.0.0.1:8028"
    api_key: str = ""
    model: str = ""
    voice: str = ""
    ref_audio: str = ""
    ref_text: str = ""
    source_sample_rate: int = 48_000
    timeout_seconds: float = 180.0
    streaming: bool = True
    # VoxCPM2 扩展参数；空=不发送。
    seed: str = ""
    max_steps: str = ""
    inference_timesteps: str = ""
    cfg_value: str = ""
    temperature: str = ""


@dataclass(frozen=True, slots=True)
class SessionSettings:
    machine_id: str = "cat-306-demo"
    operator_id: str = "voice-operator"
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VadSettings = field(default_factory=VadSettings)
    wake: WakeSettings = field(default_factory=WakeSettings)
    asr: AsrSettings = field(default_factory=AsrSettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    session: SessionSettings = field(default_factory=SessionSettings)
    source_path: Path | None = None

    def with_overrides(self, **sections: Any) -> "VoiceConfig":
        """返回替换若干配置段的副本（CLI 参数覆盖用）。"""

        return replace(self, **{key: value for key, value in sections.items() if value is not None})


def default_config_path() -> Path | None:
    candidates = [
        _env("CAT_VOICE_CONFIG"),
        "runtime/voice-config.json",
        "config/voice-config.json",
    ]
    for name in candidates:
        if name:
            path = Path(name)
            if path.is_file():
                return path
    return None


def load_voice_config(path: Path | str | None = None) -> VoiceConfig:
    """读取语音配置；文件缺失或格式错误时回退默认值。"""

    selected = Path(path) if path is not None else default_config_path()
    raw: dict[str, Any] = {}
    if selected is not None:
        try:
            value = json.loads(selected.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                raw = value
        except (OSError, json.JSONDecodeError):
            raw = {}

    audio = _section(raw, "audio")
    vad = _section(raw, "vad")
    wake = _section(raw, "wake")
    asr = _section(raw, "asr")
    tts = _section(raw, "tts")
    session = _section(raw, "session")

    config = VoiceConfig(
        audio=AudioSettings(
            input_device=_device(
                _env("CAT_VOICE_INPUT_DEVICE") or audio.get("input_device")
            ),
            output_device=_device(
                _env("CAT_VOICE_OUTPUT_DEVICE") or audio.get("output_device")
            ),
            sample_rate=_env_int(
                "CAT_VOICE_SAMPLE_RATE", _integer(audio, "sample_rate", 16_000)
            ),
            output_sample_rate=_env_int(
                "CAT_VOICE_OUTPUT_SAMPLE_RATE",
                _integer(audio, "output_sample_rate", 24_000),
            ),
            frame_ms=_integer(audio, "frame_ms", 20),
        ),
        vad=VadSettings(
            silence_ms=_env_int(
                "CAT_VOICE_SILENCE_MS", _integer(vad, "silence_ms", 300)
            ),
            min_speech_ms=_integer(vad, "min_speech_ms", 300),
            max_utterance_s=_integer(vad, "max_utterance_s", 15),
            pre_roll_ms=_integer(vad, "pre_roll_ms", 300),
            energy_threshold=_env_float(
                "CAT_VOICE_ENERGY_THRESHOLD",
                float(_number(vad, "energy_threshold", 0.0025)),
            ),
            noise_ratio=float(_number(vad, "noise_ratio", 2.0)),
        ),
        wake=WakeSettings(
            provider=(
                _env("CAT_VOICE_WAKE_PROVIDER") or _text(wake, "provider", "sherpa_onnx")
            ),
            model=_env("CAT_VOICE_WAKE_MODEL") or _text(wake, "model", DEFAULT_WAKE_MODEL),
            word=_env("CAT_VOICE_WAKE_WORD") or _text(wake, "word", DEFAULT_WAKE_WORD),
            keywords_file=_text(wake, "keywords_file", ""),
            threshold=float(_number(wake, "threshold", 0.25)),
            keywords_score=float(_number(wake, "keywords_score", 1.0)),
            num_threads=_integer(wake, "num_threads", 2),
            ack=_text(wake, "ack", "voice").lower(),
            ack_text=_text(wake, "ack_text", "我在"),
            ack_audio=_text(wake, "ack_audio", ""),
            ack_timeout_seconds=float(_number(wake, "ack_timeout_seconds", 10.0)),
            followup_timeout_ms=_env_int(
                "CAT_VOICE_FOLLOWUP_MS",
                _integer(wake, "followup_timeout_ms", 15_000),
            ),
        ),
        asr=AsrSettings(
            base_url=(
                _env("ASR_BASE_URL") or _text(asr, "base_url", "http://127.0.0.1:8021")
            ),
            api_key=_env("ASR_API_KEY") or _text(asr, "api_key", ""),
            model=_env("ASR_MODEL") or _text(asr, "model", ""),
            language=_env("ASR_LANGUAGE") or _text(asr, "language", "zh"),
            timeout_seconds=_env_float(
                "ASR_TIMEOUT_SECONDS", float(_number(asr, "timeout_seconds", 60.0))
            ),
        ),
        tts=TtsSettings(
            base_url=(
                _env("TTS_BASE_URL") or _text(tts, "base_url", "http://127.0.0.1:8028")
            ),
            api_key=_env("TTS_API_KEY") or _env("VOXCPM_API_KEY") or _text(tts, "api_key", ""),
            model=_env("VOXCPM_MODEL") or _text(tts, "model", ""),
            voice=_env("VOXCPM_VOICE") or _text(tts, "voice", ""),
            ref_audio=_env("VOXCPM_REF_AUDIO") or _text(tts, "ref_audio", ""),
            ref_text=_env("VOXCPM_REF_TEXT") or _text(tts, "ref_text", ""),
            source_sample_rate=_env_int(
                "VOXCPM_SOURCE_SAMPLE_RATE",
                _integer(tts, "source_sample_rate", 48_000),
            ),
            timeout_seconds=_env_float(
                "VOXCPM_TIMEOUT_SECONDS",
                float(_number(tts, "timeout_seconds", 180.0)),
            ),
            streaming=_env_bool("VOXCPM_STREAMING", _boolean(tts, "streaming", True)),
            seed=_env("VOXCPM_SEED") or _text(tts, "seed", ""),
            max_steps=_env("VOXCPM_MAX_STEPS") or _text(tts, "max_steps", ""),
            inference_timesteps=_env("VOXCPM_INFERENCE_TIMESTEPS")
            or _text(tts, "inference_timesteps", ""),
            cfg_value=_env("VOXCPM_CFG_VALUE") or _text(tts, "cfg_value", ""),
            temperature=_env("VOXCPM_TEMPERATURE") or _text(tts, "temperature", ""),
        ),
        session=SessionSettings(
            machine_id=_env("CAT_MACHINE_ID") or _text(session, "machine_id", "cat-306-demo"),
            operator_id=_env("CAT_OPERATOR_ID")
            or _text(session, "operator_id", "voice-operator"),
            session_id=_env("CAT_SESSION_ID") or _text(session, "session_id", ""),
        ),
        source_path=selected,
    )
    return config


def resolve_path(value: str, *, base: Path | None = None) -> Path:
    """相对路径按当前工作目录解析；``base`` 仅用于调用方显式提供基准目录。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ((base or Path.cwd()) / path).resolve()


__all__ = [
    "AsrSettings",
    "AudioSettings",
    "DEFAULT_WAKE_MODEL",
    "DEFAULT_WAKE_WORD",
    "SessionSettings",
    "TtsSettings",
    "VadSettings",
    "VoiceConfig",
    "WakeSettings",
    "default_config_path",
    "load_voice_config",
    "resolve_path",
]
