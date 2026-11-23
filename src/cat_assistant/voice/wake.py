"""本地唤醒词：sherpa-onnx 流式 KWS。

唤醒模型按拼音 token 匹配；中文唤醒词通过 pypinyin 转成声母/韵母 token。
``keywords_file`` 非空时优先使用（每行 ``token ... @显示文本``）。
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from cat_assistant.voice.config import WakeSettings, resolve_path
from cat_assistant.voice.pcm import AudioFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WakeEvent:
    detected: bool
    score: float = 0.0
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or "wake"


class WakeDetector(Protocol):
    name: str

    def detect(self, frame: AudioFrame) -> WakeEvent: ...

    def flush(self, *, duration_ms: int = 1000) -> WakeEvent: ...

    def reset(self) -> None: ...


class NullWakeDetector:
    """不做唤醒检测：常开麦克风场景下直接进入 VAD。"""

    name = "none"

    def detect(self, frame: AudioFrame) -> WakeEvent:
        del frame
        return WakeEvent(False)

    def flush(self, *, duration_ms: int = 1000) -> WakeEvent:
        del duration_ms
        return WakeEvent(False)

    def reset(self) -> None:
        return


class SherpaOnnxWakeDetector:
    """sherpa-onnx 流式 KWS；命中时 score 固定为 1.0。"""

    name = "sherpa_onnx"

    def __init__(self, config: WakeSettings):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sherpa-onnx，无法使用本地唤醒；请执行 "
                "python -m pip install 'equip-ai-agent[voice-wake]'，"
                "或加 --no-wake 关闭唤醒"
            ) from exc

        if not config.model:
            raise ValueError("sherpa_onnx 唤醒需要配置 wake.model 模型目录")
        self.model_dir = resolve_path(config.model)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"唤醒模型目录不存在：{self.model_dir}；"
                "先运行 ./scripts/download_kws_model.sh 下载模型"
            )

        tokens_path = self.model_dir / "tokens.txt"
        if not tokens_path.is_file():
            raise FileNotFoundError(f"唤醒模型缺少 tokens.txt：{self.model_dir}")
        self._tokens = _read_tokens(tokens_path)
        self._encoder, self._decoder, self._joiner = _model_files(self.model_dir)

        self._keywords_file, self._generated_keywords = _keywords_file(config, self._tokens)
        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens_path),
            encoder=str(self._encoder),
            decoder=str(self._decoder),
            joiner=str(self._joiner),
            keywords_file=str(self._keywords_file),
            num_threads=max(1, int(config.num_threads)),
            max_active_paths=4,
            keywords_score=float(config.keywords_score),
            keywords_threshold=float(config.threshold),
            num_trailing_blanks=1,
            provider="cpu",
        )
        self._stream = None
        self._stream_sample_rate = 0

    def detect(self, frame: AudioFrame) -> WakeEvent:
        if not frame.pcm:
            return WakeEvent(False)
        if len(frame.pcm) % 2:
            raise ValueError("唤醒输入必须是完整的 s16le 采样")
        if self._stream is None or self._stream_sample_rate != frame.sample_rate:
            self._stream = self._kws.create_stream()
            self._stream_sample_rate = frame.sample_rate

        samples = np.frombuffer(frame.pcm, dtype="<i2").astype(np.float32) / 32768.0
        self._stream.accept_waveform(frame.sample_rate, samples)
        return self._decode_ready()

    def _decode_ready(self) -> WakeEvent:
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)
            result = self._kws.get_result(self._stream)
            if result:
                # 官方要求命中后立即 reset，避免同一段音频重复触发。
                self._kws.reset_stream(self._stream)
                return WakeEvent(True, 1.0, result)
        return WakeEvent(False)

    def flush(self, *, duration_ms: int = 1000) -> WakeEvent:
        """用尾部静音补齐特征窗口，处理文件结束前尚未解码的音频。"""

        if self._stream is None or not self._stream_sample_rate:
            return WakeEvent(False)
        count = max(1, int(self._stream_sample_rate * duration_ms / 1000))
        self._stream.accept_waveform(
            self._stream_sample_rate, np.zeros(count, dtype=np.float32)
        )
        self._stream.input_finished()
        return self._decode_ready()

    def reset(self) -> None:
        if self._stream is not None:
            self._kws.reset_stream(self._stream)

    def close(self) -> None:
        self.reset()
        path, self._generated_keywords = self._generated_keywords, None
        if path is not None:
            try:
                path.unlink()
            except OSError:
                pass


def build_wake(config: WakeSettings) -> WakeDetector:
    name = (config.provider or "none").strip().lower()
    if name in ("none", "off", "disabled", ""):
        return NullWakeDetector()
    if name in ("sherpa_onnx", "sherpa-onnx", "sherpaonnx", "kws"):
        return SherpaOnnxWakeDetector(config)
    raise ValueError(
        f"未知 wake provider：{name}（精简版只支持 sherpa_onnx/none）"
    )


def _read_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2:
            tokens.add(parts[0])
    return tokens


def _model_files(model_dir: Path) -> tuple[Path, Path, Path]:
    """选择官方示例使用的 epoch-12 fp32 模型；其他命名按前缀回退。"""

    def pick(prefix: str) -> Path:
        preferred = model_dir / f"{prefix}-epoch-12-avg-2-chunk-16-left-64.onnx"
        if preferred.is_file():
            return preferred
        candidates = sorted(model_dir.glob(f"{prefix}*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"唤醒模型目录缺少 {prefix}*.onnx：{model_dir}")
        non_int8 = [path for path in candidates if ".int8." not in path.name]
        return (non_int8 or candidates)[0]

    return pick("encoder"), pick("decoder"), pick("joiner")


def _keywords_file(config: WakeSettings, tokens: set[str]) -> tuple[Path, Path | None]:
    if config.keywords_file:
        path = resolve_path(config.keywords_file)
        if not path.is_file():
            raise FileNotFoundError(f"唤醒关键词文件不存在：{path}")
        return path, None

    word = (config.word or "").strip()
    if not word:
        raise ValueError("sherpa_onnx 唤醒需要配置 wake.word 或 wake.keywords_file")
    line = _keyword_line(word, tokens)
    fd, raw_path = tempfile.mkstemp(prefix="cat-kws-", suffix=".txt")
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path, path


def _keyword_line(word: str, tokens: set[str]) -> str:
    """把中文唤醒词转成 sherpa-onnx KWS 的拼音 token 行。"""

    try:
        from pypinyin import Style, pinyin
    except ImportError as exc:
        raise RuntimeError(
            "按 wake.word 自动生成关键词需要 pypinyin；请执行 "
            "python -m pip install pypinyin，或直接配置 wake.keywords_file"
        ) from exc

    initials = pinyin(
        word,
        style=Style.INITIALS,
        strict=False,
        neutral_tone_with_five=True,
        errors="ignore",
    )
    finals = pinyin(
        word,
        style=Style.FINALS_TONE,
        strict=False,
        neutral_tone_with_five=True,
        errors="ignore",
    )
    if len(initials) != len(finals) or not initials:
        raise ValueError(f"无法把唤醒词转换为拼音：{word!r}")

    parts: list[str] = []
    for initial, final in zip(initials, finals):
        for token in (initial[0], final[0]):
            if not token:
                continue
            if token not in tokens:
                raise ValueError(
                    f"唤醒词 {word!r} 的拼音 token {token!r} 不在模型 tokens.txt 中；"
                    "请检查 wake.word 或改用 wake.keywords_file"
                )
            parts.append(token)
    if not parts:
        raise ValueError(f"唤醒词没有可用拼音 token：{word!r}")

    label = re.sub(r"[\s，,。.!！?？、；;：:]+", "", word) or word
    return " ".join(parts) + f" @{label}"


__all__ = [
    "NullWakeDetector",
    "SherpaOnnxWakeDetector",
    "WakeDetector",
    "WakeEvent",
    "build_wake",
]
