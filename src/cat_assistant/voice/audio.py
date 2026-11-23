"""音频输入与输出：sounddevice 麦克风/扬声器、WAV 文件与内存端。

sounddevice 只在真正打开设备时导入，因此无音频设备的机器仍可使用
WAV 文件模式和调试 UI 的网页录音。
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import AsyncIterator, Protocol

from cat_assistant.voice.pcm import (
    AudioFrame,
    iter_pcm_frames,
    read_wav,
    resample_s16,
    write_wav,
)


class AudioSink(Protocol):
    async def play(self, pcm: bytes) -> None: ...

    async def stop(self) -> None: ...

    async def close(self) -> None: ...


class AudioSource(Protocol):
    async def frames(self) -> AsyncIterator[AudioFrame]: ...

    async def discard_pending(self) -> int: ...

    async def close(self) -> None: ...


class NullSink:
    async def play(self, pcm: bytes) -> None:
        del pcm

    async def stop(self) -> None:
        return

    async def close(self) -> None:
        return


class MemorySink:
    """把音频收集到内存，测试使用。"""

    def __init__(self, sample_rate: int = 24_000):
        self.sample_rate = sample_rate
        self.buffer = bytearray()

    async def play(self, pcm: bytes) -> None:
        self.buffer.extend(pcm)

    async def stop(self) -> None:
        return

    async def close(self) -> None:
        return

    @property
    def pcm(self) -> bytes:
        return bytes(self.buffer)


class WavSink:
    """把播放内容写入一个 WAV 文件。"""

    def __init__(self, path: str | Path, sample_rate: int = 24_000):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.buffer = bytearray()

    async def play(self, pcm: bytes) -> None:
        self.buffer.extend(pcm)

    async def stop(self) -> None:
        return

    async def close(self) -> None:
        write_wav(self.path, bytes(self.buffer), self.sample_rate)


class SpeakerSink:
    """sounddevice 扬声器输出；整场会话复用同一个输出流。"""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        sample_rate: int = 24_000,
        blocksize_ms: int = 100,
    ):
        self.device = _device(device)
        self.sample_rate = int(sample_rate)
        self.blocksize_ms = max(1, int(blocksize_ms))
        self._stream = None
        self._write_task: asyncio.Task | None = None

    def _open(self):
        import sounddevice as sd

        stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
            blocksize=int(self.sample_rate * self.blocksize_ms / 1000),
            latency="low",
        )
        stream.start()
        return stream

    async def play(self, pcm: bytes) -> None:
        if not pcm:
            return
        try:
            if self._stream is None:
                self._stream = await asyncio.to_thread(self._open)
            stream = self._stream
            # stop() 用 abort() 打断后流仍可复用，但需要补一次 start()。
            if getattr(stream, "active", True) is False:
                await asyncio.to_thread(stream.start)
            write_task = asyncio.create_task(asyncio.to_thread(stream.write, pcm))
            self._write_task = write_task
            try:
                await asyncio.shield(write_task)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(stream.abort)
                raise
            finally:
                if self._write_task is write_task:
                    self._write_task = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stream, self._stream = self._stream, None
            if stream is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(stream.close)
            raise RuntimeError(f"扬声器播放失败：{exc}") from exc

    async def stop(self) -> None:
        stream = self._stream
        if stream is None:
            return
        with contextlib.suppress(Exception):
            await asyncio.to_thread(stream.abort)
        write_task = self._write_task
        if write_task is not None and not write_task.done():
            with contextlib.suppress(Exception):
                await asyncio.shield(write_task)

    async def close(self) -> None:
        await self.stop()
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stream.close)


class WavFileSource:
    """从 WAV 文件读取；realtime=True 时按真实时间节奏输出。"""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        realtime: bool = False,
    ):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.realtime = realtime

    async def frames(self) -> AsyncIterator[AudioFrame]:
        pcm, rate = read_wav(self.path)
        if rate != self.sample_rate:
            pcm = resample_s16(pcm, rate, self.sample_rate)
        timestamp = 0
        for chunk in iter_pcm_frames(pcm, self.sample_rate, self.frame_ms):
            yield AudioFrame(chunk, self.sample_rate, timestamp)
            timestamp += self.frame_ms
            if self.realtime:
                await asyncio.sleep(self.frame_ms / 1000)

    async def discard_pending(self) -> int:
        return 0

    async def close(self) -> None:
        return


class MicSource:
    """sounddevice 麦克风输入；回调线程把 PCM 帧放入 asyncio 队列。"""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        queue_size: int = 400,
    ):
        self.device = _device(device)
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.queue_size = max(10, int(queue_size))
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        self._stream = None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sounddevice，无法使用麦克风；请执行 "
                "python -m pip install 'equip-ai-agent[voice]'"
            ) from exc

        loop = asyncio.get_running_loop()
        blocksize = int(self.sample_rate * self.frame_ms / 1000)

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info, status
            try:
                loop.call_soon_threadsafe(self._put, bytes(indata))
            except RuntimeError:
                return

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=blocksize,
                device=self.device,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(
                f"无法打开麦克风设备：{exc}；用 --list-devices 查看可用设备"
            ) from exc

        timestamp = 0
        try:
            while True:
                pcm = await self._queue.get()
                if pcm is None:
                    return
                yield AudioFrame(pcm, self.sample_rate, timestamp)
                timestamp += self.frame_ms
        finally:
            self._close_stream()

    def _put(self, payload: bytes) -> None:
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(payload)

    async def discard_pending(self) -> int:
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            dropped += 1
        return dropped

    def _close_stream(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
            with contextlib.suppress(Exception):
                self._stream.close()
            self._stream = None

    async def close(self) -> None:
        self._close_stream()
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)


def list_devices() -> list[dict]:
    """返回 sounddevice 设备列表；无依赖或无声卡时抛出明确错误。"""

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "未安装 sounddevice；请执行 python -m pip install 'equip-ai-agent[voice]'"
        ) from exc
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"无法枚举音频设备：{exc}") from exc
    result = []
    for index, item in enumerate(devices):
        result.append(
            {
                "index": index,
                "name": item.get("name", ""),
                "inputs": int(item.get("max_input_channels", 0)),
                "outputs": int(item.get("max_output_channels", 0)),
                "sample_rate": int(item.get("default_samplerate", 0)),
            }
        )
    return result


def _device(value):
    if value in (None, "", "default"):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


__all__ = [
    "AudioSink",
    "AudioSource",
    "MemorySink",
    "MicSource",
    "NullSink",
    "SpeakerSink",
    "WavFileSource",
    "WavSink",
    "list_devices",
]
