"""TTS 播报：分句合成与播放解耦，一句话还在播放时下一句已开始合成。"""

from __future__ import annotations

import asyncio
import logging
import time

from cat_assistant.voice.audio import AudioSink
from cat_assistant.voice.tts import SentenceSplitter, VoxCpmClient

logger = logging.getLogger(__name__)


class SpeechPlayer:
    def __init__(
        self,
        tts: VoxCpmClient,
        sink: AudioSink,
        *,
        output_sample_rate: int = 24_000,
    ):
        self.tts = tts
        self.sink = sink
        self.output_sample_rate = int(output_sample_rate)
        self.splitter = SentenceSplitter()
        self.last_error = ""

    async def say(self, text: str) -> bool:
        """合成并播放整段文本；失败时记录 ``last_error`` 但不抛出。"""

        self.last_error = ""
        sentences = self.splitter.split(text or "")
        self.splitter.reset()
        if not sentences:
            return False

        audio_queue: asyncio.Queue = asyncio.Queue()
        done = object()

        async def synth() -> None:
            try:
                for sentence in sentences:
                    async for chunk in self.tts.stream(sentence):
                        if chunk:
                            await audio_queue.put(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"TTS 合成失败：{type(exc).__name__}: {exc}"
                logger.warning("%s", self.last_error)
            finally:
                await audio_queue.put(done)

        async def play() -> None:
            while True:
                chunk = await audio_queue.get()
                if chunk is done:
                    return
                try:
                    await self.sink.play(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.last_error = f"TTS 播放失败：{type(exc).__name__}: {exc}"
                    logger.warning("%s", self.last_error)
                    return

        synth_task = asyncio.create_task(synth(), name="voice-tts-synth")
        play_task = asyncio.create_task(play(), name="voice-tts-play")
        try:
            await asyncio.gather(synth_task, play_task)
        except asyncio.CancelledError:
            synth_task.cancel()
            play_task.cancel()
            await asyncio.gather(synth_task, play_task, return_exceptions=True)
            raise
        except Exception as exc:  # 队列/任务异常，保持语音循环存活
            self.last_error = f"TTS 播报异常：{type(exc).__name__}: {exc}"
            logger.warning("%s", self.last_error)
        return not self.last_error

    async def play_pcm(self, pcm: bytes) -> None:
        """直接播放一段已是播放采样率的 PCM（唤醒确认音等）。"""

        if not pcm:
            return
        started = time.perf_counter()
        try:
            await self.sink.play(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("播放提示音失败：%s", exc)
        finally:
            logger.debug("提示音播放耗时 %.0fms", (time.perf_counter() - started) * 1000)

    async def close(self) -> None:
        self.splitter.reset()
        try:
            await self.sink.close()
        except Exception as exc:
            logger.warning("关闭音频输出失败：%s", exc)


__all__ = ["SpeechPlayer"]
