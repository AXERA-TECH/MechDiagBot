from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

try:
    import numpy as np

    from cat_assistant.domain.models import AssistantResponse, ResponseCategory, Utterance
    from cat_assistant.voice.asr import QwenAsrClient
    from cat_assistant.voice.audio import NullSink, WavFileSource, WavSink
    from cat_assistant.voice.config import (
        AsrSettings,
        AudioSettings,
        SessionSettings,
        TtsSettings,
        VoiceConfig,
        WakeSettings,
        load_voice_config,
    )
    from cat_assistant.voice.http import VoiceServiceError, endpoint_for
    from cat_assistant.voice.loop import VoiceLoop
    from cat_assistant.voice.pcm import (
        AudioFrame,
        StreamResampler,
        VoiceUtterance,
        decode_wav_bytes,
        pcm_duration_ms,
        read_wav,
        resample_s16,
        silence,
        wav_bytes,
        write_wav,
    )
    from cat_assistant.voice.tts import SentenceSplitter, VoxCpmClient, clean_for_speech
    from cat_assistant.voice.vad import EnergyVad, build_segmenter
    from cat_assistant.voice.wake import NullWakeDetector
    from cat_assistant.voice.web import WebVoiceService

    VOICE_READY = True
except ImportError:  # numpy 是可选语音依赖；缺失时跳过语音测试。
    np = None
    VOICE_READY = False


def _serve(responder):
    """启动本地 HTTP 服务，返回 (base_url, records, server, thread)。"""

    records: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length)
            records.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": body,
                }
            )
            status, content_type, payload = responder(records[-1])
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return base, records, server, thread


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class PcmTests(unittest.TestCase):
    def _samples(self, count: int = 480) -> bytes:
        return np.arange(count, dtype="<i2").tobytes()

    def test_resample_decimates_integer_ratio(self):
        pcm = self._samples(480)
        resampled = resample_s16(pcm, 48_000, 24_000)
        self.assertEqual(len(resampled), 240 * 2)
        self.assertEqual(resampled, np.frombuffer(pcm, dtype="<i2")[::2].tobytes())
        upsampled = resample_s16(resampled, 24_000, 48_000)
        self.assertGreater(len(upsampled), len(resampled))

    def test_stream_resampler_keeps_phase_across_chunks(self):
        pcm = self._samples(480)
        expected = resample_s16(pcm, 48_000, 24_000)
        resampler = StreamResampler(48_000, 24_000)
        out = b""
        # 奇数字节 / 非采样对齐的分块边界都要能续上。
        for index in range(0, len(pcm), 7):
            out += resampler.push(pcm[index : index + 7])
        out += resampler.flush()
        self.assertEqual(out, expected)

    def test_stream_resampler_flushes_odd_tail(self):
        resampler = StreamResampler(48_000, 16_000)
        self.assertEqual(resampler.push(b"\x01\x02\x03"), b"")
        self.assertEqual(resampler.push(b"\x04\x05\x06\x07\x08\x09"), b"\x01\x02")
        self.assertEqual(resampler.flush(), b"\x07\x08")
        self.assertEqual(resampler.flush(), b"")

    def test_wav_roundtrip_and_resampling(self):
        pcm = self._samples(480)
        encoded = wav_bytes(pcm, 48_000)
        decoded, rate = decode_wav_bytes(encoded, 24_000)
        self.assertEqual(rate, 24_000)
        self.assertEqual(decoded, resample_s16(pcm, 48_000, 24_000))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            write_wav(path, pcm, 48_000)
            kept, kept_rate = read_wav(path)
            self.assertEqual(kept_rate, 48_000)
            self.assertEqual(kept, pcm)
            converted, converted_rate = read_wav(path, target_rate=24_000)
            self.assertEqual(converted_rate, 24_000)
            self.assertEqual(converted, resample_s16(pcm, 48_000, 24_000))

    def test_duration_and_frame_bytes(self):
        pcm = silence(20, 16_000)
        self.assertEqual(pcm_duration_ms(pcm, 16_000), 20)


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class SplitterTests(unittest.TestCase):
    def test_split_sentences_and_incremental_push(self):
        splitter = SentenceSplitter()
        self.assertEqual(splitter.push("你好。今天设备"), ["你好。"])
        self.assertEqual(splitter.push("运行正常！"), ["今天设备运行正常！"])
        self.assertEqual(splitter.flush(), [])

    def test_does_not_split_inside_quotes(self):
        text = "他说：“请立即停机。”然后等待。"
        self.assertEqual(SentenceSplitter().split(text), [text])

    def test_long_sentence_breaks_at_soft_punctuation(self):
        splitter = SentenceSplitter(max_chars=12)
        parts = splitter.split("第一段内容比较长，第二段内容也比较长。")
        self.assertGreaterEqual(len(parts), 2)
        for part in parts:
            self.assertLessEqual(len(part), 12)

    def test_clean_for_speech_removes_markup(self):
        self.assertEqual(
            clean_for_speech("**启动** [文档](https://example.com) 完成"),
            "启动 文档 完成",
        )


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class VadTests(unittest.TestCase):
    def _frame(self, loud: bool, timestamp_ms: int) -> AudioFrame:
        values = np.full(320, 12_000 if loud else 0, dtype="<i2")
        return AudioFrame(values.tobytes(), 16_000, timestamp_ms)

    def test_energy_vad_adapts_to_noise(self):
        vad = EnergyVad(floor=0.01, noise_ratio=2.0)
        self.assertFalse(vad.is_speech(silence(20, 16_000), 16_000))
        self.assertAlmostEqual(vad.gate, 0.01, places=6)
        self.assertTrue(vad.is_speech(self._frame(True, 0).pcm, 16_000))

    def test_segmenter_emits_utterance_after_silence(self):
        segmenter = build_segmenter(
            silence_ms=100,
            min_speech_ms=100,
            pre_roll_ms=60,
            energy_threshold=0.01,
        )
        utterance = None
        for index in range(5):
            self.assertIsNone(segmenter.push(self._frame(False, index * 20)))
        for index in range(5, 10):
            self.assertIsNone(segmenter.push(self._frame(True, index * 20)))
        for index in range(10, 15):
            utterance = segmenter.push(self._frame(False, index * 20))
            if utterance is not None:
                break
        self.assertIsNotNone(utterance)
        self.assertEqual(utterance.sample_rate, 16_000)
        self.assertGreaterEqual(utterance.duration_ms, 200)
        self.assertGreater(utterance.end_ms, utterance.start_ms)

    def test_segmenter_flush_finishes_active_speech(self):
        segmenter = build_segmenter(min_speech_ms=60, energy_threshold=0.01)
        for index in range(5):
            segmenter.push(self._frame(True, index * 20))
        utterance = segmenter.flush()
        self.assertIsNotNone(utterance)
        self.assertTrue(utterance.pcm)
        self.assertIsNone(segmenter.flush())


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class ConfigTests(unittest.TestCase):
    def test_loads_sections_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.json"
            path.write_text(
                json.dumps(
                    {
                        "audio": {
                            "input_device": "3",
                            "output_sample_rate": 22_050,
                        },
                        "vad": {"silence_ms": 500},
                        "wake": {"provider": "none", "word": "嘿小安"},
                        "asr": {"base_url": "http://asr:9000/v1", "language": "en"},
                        "tts": {"streaming": False, "source_sample_rate": 22_050},
                        "session": {"machine_id": "cat-999"},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                config = load_voice_config(path)
            self.assertEqual(config.audio.input_device, 3)
            self.assertEqual(config.audio.sample_rate, 16_000)
            self.assertEqual(config.audio.output_sample_rate, 22_050)
            self.assertEqual(config.vad.silence_ms, 500)
            self.assertFalse(config.wake.enabled)
            self.assertEqual(config.wake.word, "嘿小安")
            self.assertEqual(config.asr.base_url, "http://asr:9000/v1")
            self.assertEqual(config.asr.language, "en")
            self.assertFalse(config.tts.streaming)
            self.assertEqual(config.tts.source_sample_rate, 22_050)
            self.assertEqual(config.session.machine_id, "cat-999")
            self.assertEqual(config.source_path, path)

    def test_environment_overrides_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.json"
            path.write_text(
                json.dumps({"asr": {"base_url": "http://file:1"}}),
                encoding="utf-8",
            )
            env = {
                "ASR_BASE_URL": "http://env:2/v1",
                "CAT_VOICE_SAMPLE_RATE": "48000",
                "CAT_VOICE_WAKE_PROVIDER": "none",
                "VOXCPM_STREAMING": "false",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                config = load_voice_config(path)
            self.assertEqual(config.asr.base_url, "http://env:2/v1")
            self.assertEqual(config.audio.sample_rate, 48_000)
            self.assertFalse(config.wake.enabled)
            self.assertFalse(config.tts.streaming)

    def test_missing_file_uses_defaults_and_overrides(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = load_voice_config("/nonexistent/voice.json")
        self.assertEqual(config.audio.sample_rate, 16_000)
        self.assertEqual(config.asr.base_url, "http://127.0.0.1:8021")
        self.assertEqual(config.tts.base_url, "http://127.0.0.1:8028")
        replaced = config.with_overrides(asr=AsrSettings(base_url="http://other:1"))
        self.assertEqual(replaced.asr.base_url, "http://other:1")
        self.assertEqual(config.asr.base_url, "http://127.0.0.1:8021")

    def test_endpoint_for_normalizes_base_paths(self):
        path = "/v1/audio/transcriptions"
        self.assertEqual(endpoint_for("http://x:1", path), f"http://x:1{path}")
        self.assertEqual(endpoint_for("http://x:1/v1", path), f"http://x:1{path}")
        self.assertEqual(endpoint_for(f"http://x:1{path}", path), f"http://x:1{path}")
        with self.assertRaises(ValueError):
            endpoint_for("x:1", path)


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class AsrClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_posts_multipart_wav(self):
        base, records, server, thread = _serve(
            lambda record: (
                200,
                "application/json",
                json.dumps({"text": " 启动三号泵 ", "language": "zh"}).encode(),
            )
        )
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        client = QwenAsrClient(
            AsrSettings(
                base_url=base,
                api_key="secret",
                model="qwen3-asr",
                language="zh",
            )
        )
        pcm = silence(40, 16_000)
        text = await client.transcribe(pcm, 16_000)

        self.assertEqual(text, "启动三号泵")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["path"], "/v1/audio/transcriptions")
        headers = {key.lower(): value for key, value in record["headers"].items()}
        self.assertEqual(headers["authorization"], "Bearer secret")
        self.assertTrue(headers["content-type"].startswith("multipart/form-data; boundary="))
        body = record["body"]
        self.assertIn(b'name="file"; filename="utterance.wav"', body)
        self.assertIn(b"Content-Type: audio/wav", body)
        self.assertIn(b"RIFF", body)
        for field in (b"response_format", b"json", b"language", b"zh", b"model", b"qwen3-asr"):
            self.assertIn(field, body)

    async def test_transcribe_rejects_invalid_pcm_and_surfaces_http_errors(self):
        client = QwenAsrClient(AsrSettings(base_url="http://127.0.0.1:1"))
        with self.assertRaises(ValueError):
            await client.transcribe(b"\x00", 16_000)
        self.assertEqual(await client.transcribe(b"", 16_000), "")

        base, _, server, thread = _serve(
            lambda record: (500, "application/json", b'{"error": "boom"}')
        )
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        failing = QwenAsrClient(AsrSettings(base_url=base))
        with self.assertRaises(VoiceServiceError) as caught:
            await failing.transcribe(silence(20, 16_000), 16_000)
        self.assertIn("HTTP 500", str(caught.exception))
        self.assertIn("boom", str(caught.exception))


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class TtsClientTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self, base: str, **overrides) -> TtsSettings:
        values = {
            "base_url": base,
            "model": "voxcpm2",
            "voice": "default",
            "ref_text": "参考音色",
            "source_sample_rate": 48_000,
        }
        values.update(overrides)
        return TtsSettings(**values)

    async def test_stream_resamples_raw_pcm_to_output_rate(self):
        source_pcm = np.arange(4800, dtype="<i2").tobytes()
        base, records, server, thread = _serve(
            lambda record: (200, "audio/pcm", source_pcm)
        )
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        client = VoxCpmClient(self._settings(base), output_sample_rate=24_000)
        chunks = [chunk async for chunk in client.stream("你好。")]
        pcm = b"".join(chunks)

        self.assertEqual(pcm, resample_s16(source_pcm, 48_000, 24_000))
        payload = json.loads(records[0]["body"])
        self.assertEqual(payload["input"], "你好。")
        self.assertEqual(payload["response_format"], "pcm")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["model"], "voxcpm2")
        self.assertEqual(payload["voice"], "default")
        self.assertEqual(payload["ref_text"], "参考音色")

    async def test_stream_falls_back_to_wav_body(self):
        wav_source = wav_bytes(np.zeros(480, dtype="<i2").tobytes(), 48_000)
        base, _, server, thread = _serve(
            lambda record: (200, "audio/wav", wav_source)
        )
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        client = VoxCpmClient(self._settings(base), output_sample_rate=24_000)
        pcm = await client.synthesize("你好。")
        self.assertEqual(pcm, b"\x00\x00" * 240)

    async def test_non_streaming_requests_wav_and_decodes_it(self):
        wav_source = wav_bytes(np.zeros(480, dtype="<i2").tobytes(), 48_000)
        base, records, server, thread = _serve(
            lambda record: (200, "audio/wav", wav_source)
        )
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        client = VoxCpmClient(
            self._settings(base, streaming=False), output_sample_rate=24_000
        )
        pcm = await client.synthesize("你好。")
        self.assertEqual(pcm, b"\x00\x00" * 240)
        payload = json.loads(records[0]["body"])
        self.assertEqual(payload["response_format"], "wav")
        self.assertNotIn("stream", payload)

    async def test_json_error_is_reported(self):
        base, _, server, thread = _serve(
            lambda record: (
                200,
                "application/json",
                b'{"error": {"message": "model not loaded"}}',
            )
        )
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        client = VoxCpmClient(self._settings(base), output_sample_rate=24_000)
        with self.assertRaises(VoiceServiceError) as caught:
            await client.synthesize("你好。")
        self.assertIn("model not loaded", str(caught.exception))


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class WebVoiceServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_resamples_to_16k(self):
        config = VoiceConfig()
        service = WebVoiceService(config)
        calls: list[tuple[bytes, int]] = []

        class FakeAsr:
            async def transcribe(self, pcm, sample_rate=16_000):
                calls.append((pcm, sample_rate))
                return "启动三号泵"

        service.asr = FakeAsr()
        pcm = np.zeros(800, dtype="<i2").tobytes()
        text = await service.transcribe(pcm, 8_000)

        self.assertEqual(text, "启动三号泵")
        self.assertEqual(calls[0][1], 16_000)
        self.assertEqual(len(calls[0][0]), 1600 * 2)

    async def test_speak_wav_cleans_markup_and_wraps_wav(self):
        config = VoiceConfig(audio=AudioSettings(output_sample_rate=24_000))
        service = WebVoiceService(config)
        spoken: list[str] = []

        class FakeTts:
            async def synthesize(self, text):
                spoken.append(text)
                return b"\x00\x00" * 240

        service.tts = FakeTts()
        data = await service.speak_wav("**你好**")
        self.assertEqual(spoken, ["你好"])
        with wave.open(io.BytesIO(data), "rb") as handle:
            self.assertEqual(handle.getframerate(), 24_000)
            self.assertEqual(handle.getnframes(), 240)
            self.assertEqual(handle.getnchannels(), 1)
        self.assertEqual(await service.speak_wav("**"), b"")

    async def test_stream_pcm_cleans_markup_and_skips_empty_chunks(self):
        config = VoiceConfig(audio=AudioSettings(output_sample_rate=24_000))
        service = WebVoiceService(config)
        spoken: list[str] = []

        class FakeTts:
            async def stream(self, text):
                spoken.append(text)
                yield b"\x01\x02" * 4
                yield b""
                yield b"\x03\x04" * 2

        service.tts = FakeTts()
        chunks = [chunk async for chunk in service.stream_pcm("**你好**")]
        self.assertEqual(spoken, ["你好"])
        self.assertEqual(chunks, [b"\x01\x02" * 4, b"\x03\x04" * 2])
        self.assertEqual(
            [chunk async for chunk in service.stream_pcm("**")],
            [],
        )


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class VoiceLoopTests(unittest.IsolatedAsyncioTestCase):
    class FakeSource:
        def __init__(self):
            self.discarded = 0
            self.closed = False

        async def frames(self):
            if False:  # pragma: no cover - 空异步生成器
                yield AudioFrame(b"", 16_000, 0)

        async def discard_pending(self):
            self.discarded += 1

        async def close(self):
            self.closed = True

    class FakeSink:
        def __init__(self):
            self.chunks: list[bytes] = []
            self.closed = False

        async def play(self, pcm):
            self.chunks.append(pcm)

        async def abort(self):
            return

        async def close(self):
            self.closed = True

    class FakeAsr:
        def __init__(self, text: str):
            self.text = text
            self.calls: list[tuple[bytes, int]] = []
            self.closed = False

        async def warmup(self):
            return

        async def transcribe(self, pcm, sample_rate=16_000):
            self.calls.append((pcm, sample_rate))
            return self.text

        async def close(self):
            self.closed = True

    class FakeTts:
        def __init__(self, pcm: bytes = b"\x01\x02" * 240):
            self.pcm = pcm
            self.texts: list[str] = []
            self.closed = False

        async def warmup(self):
            return

        async def synthesize(self, text):
            self.texts.append(text)
            return self.pcm

        async def stream(self, text):
            self.texts.append(text)
            yield self.pcm

        async def close(self):
            self.closed = True

    class FakeApp:
        def __init__(self, reply: str = "已启动三号冷却泵。"):
            self.reply = reply
            self.utterances: list[Utterance] = []
            self.shutdown_called = False

        async def handle(self, utterance: Utterance) -> AssistantResponse:
            self.utterances.append(utterance)
            return AssistantResponse(text=self.reply, category=ResponseCategory.ACTION_COMPLETED)

        async def shutdown(self):
            self.shutdown_called = True

    def _loop(self, *, asr_text="启动三号泵", reply="已启动三号冷却泵。"):
        config = VoiceConfig(
            wake=WakeSettings(provider="none"),
            session=SessionSettings(
                machine_id="cat-306-demo",
                operator_id="voice-operator",
                session_id="voice-session",
            ),
        )
        source = self.FakeSource()
        sink = self.FakeSink()
        asr = self.FakeAsr(asr_text)
        tts = self.FakeTts()
        app = self.FakeApp(reply)
        loop = VoiceLoop(
            config,
            app,
            asr=asr,
            tts=tts,
            source=source,
            sink=sink,
            wake=NullWakeDetector(),
        )
        return loop, app, asr, tts, source, sink

    async def test_handle_utterance_runs_agent_and_speaks_reply(self):
        loop, app, asr, tts, source, sink = self._loop()
        utterance = VoiceUtterance(
            pcm=silence(400, 16_000), sample_rate=16_000, start_ms=0, end_ms=400
        )
        text, response = await loop.handle_utterance(utterance)

        self.assertEqual(text, "启动三号泵")
        self.assertIsNotNone(response)
        self.assertEqual(response.text, "已启动三号冷却泵。")
        self.assertEqual(len(asr.calls), 1)
        self.assertEqual(app.utterances[0].text, "启动三号泵")
        self.assertEqual(app.utterances[0].session_id, "voice-session")
        self.assertEqual(app.utterances[0].machine_id, "cat-306-demo")
        self.assertEqual(app.utterances[0].operator_id, "voice-operator")
        self.assertEqual(tts.texts, ["已启动三号冷却泵。"])
        self.assertEqual(sink.chunks, [tts.pcm])
        self.assertEqual(source.discarded, 1)
        self.assertTrue(loop._followup_active())

        await loop.close()
        self.assertTrue(source.closed)
        self.assertTrue(sink.closed)
        self.assertTrue(asr.closed)
        self.assertTrue(tts.closed)
        self.assertTrue(app.shutdown_called)

    async def test_empty_asr_result_skips_agent(self):
        loop, app, _, _, _, _ = self._loop(asr_text="")
        text, response = await loop.handle_utterance(
            VoiceUtterance(pcm=silence(400, 16_000), sample_rate=16_000)
        )
        self.assertEqual(text, "")
        self.assertIsNone(response)
        self.assertEqual(app.utterances, [])
        await loop.close()

    async def test_asr_failure_is_contained(self):
        loop, app, asr, _, _, _ = self._loop()

        async def boom(pcm, sample_rate=16_000):
            raise VoiceServiceError("服务不可用")

        asr.transcribe = boom  # type: ignore[method-assign]
        text, response = await loop.handle_utterance(
            VoiceUtterance(pcm=silence(400, 16_000), sample_rate=16_000)
        )
        self.assertEqual(text, "")
        self.assertIsNone(response)
        self.assertEqual(app.utterances, [])
        await loop.close()


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class AudioFileTests(unittest.TestCase):
    def test_wav_source_and_sink_roundtrip(self):
        async def run():
            pcm = silence(100, 16_000)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "input.wav"
                write_wav(path, pcm, 16_000)
                source = WavFileSource(path, sample_rate=8_000, frame_ms=20)
                frames = [frame async for frame in source.frames()]
                await source.close()
                self.assertTrue(frames)
                self.assertTrue(all(frame.sample_rate == 8_000 for frame in frames))

                out = Path(tmp) / "output.wav"
                sink = WavSink(out, sample_rate=16_000)
                await sink.play(pcm)
                await sink.close()
                with wave.open(str(out), "rb") as handle:
                    self.assertEqual(handle.getframerate(), 16_000)
                    self.assertEqual(handle.readframes(handle.getnframes()), pcm)

        asyncio.run(run())

    def test_null_sink_accepts_audio(self):
        async def run():
            sink = NullSink()
            await sink.play(b"\x00\x00" * 10)
            await sink.close()

        asyncio.run(run())


@unittest.skipUnless(VOICE_READY, "需要安装 numpy")
class DebugRuntimeVoiceTests(unittest.TestCase):
    def test_voice_status_transcribe_and_speak(self):
        def responder(record):
            if record["path"].endswith("/audio/transcriptions"):
                return (
                    200,
                    "application/json",
                    json.dumps({"text": "启动三号泵", "language": "zh"}).encode(),
                )
            return 200, "audio/pcm", np.zeros(480, dtype="<i2").tobytes()

        base, records, server, thread = _serve(responder)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        from cat_assistant.debug_ui import DebugRuntime

        with tempfile.TemporaryDirectory() as tmp:
            voice_config = Path(tmp) / "voice-config.json"
            voice_config.write_text(
                json.dumps(
                    {
                        "audio": {"output_sample_rate": 24_000},
                        "wake": {"provider": "none"},
                        "asr": {"base_url": base},
                        "tts": {
                            "base_url": base,
                            "streaming": True,
                            "source_sample_rate": 48_000,
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtime = DebugRuntime(
                Path(tmp) / "events.jsonl",
                voice_config_path=voice_config,
            )
            try:
                status = runtime.voice_status()
                self.assertTrue(status["available"])
                self.assertEqual(status["asr"], base)
                self.assertEqual(status["tts"], base)
                self.assertEqual(
                    runtime.voice_transcribe(silence(40, 16_000), 16_000),
                    "启动三号泵",
                )
                wav = runtime.voice_speak("**你好**")
                with wave.open(io.BytesIO(wav), "rb") as handle:
                    self.assertEqual(handle.getframerate(), 24_000)
                    self.assertEqual(handle.getnframes(), 240)
            finally:
                runtime.close()

        self.assertEqual(len(records), 2)
        payload = json.loads(records[1]["body"])
        self.assertEqual(payload["input"], "你好")
        self.assertTrue(payload["stream"])

    def test_voice_speak_stream_yields_pcm_and_playback_rate(self):
        base, records, server, thread = _serve(
            lambda record: (200, "audio/pcm", np.zeros(480, dtype="<i2").tobytes())
        )
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        from cat_assistant.debug_ui import DebugRuntime

        with tempfile.TemporaryDirectory() as tmp:
            voice_config = Path(tmp) / "voice-config.json"
            voice_config.write_text(
                json.dumps(
                    {
                        "audio": {"output_sample_rate": 24_000},
                        "wake": {"provider": "none"},
                        "tts": {
                            "base_url": base,
                            "streaming": True,
                            "source_sample_rate": 48_000,
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtime = DebugRuntime(
                Path(tmp) / "events.jsonl",
                voice_config_path=voice_config,
            )
            try:
                stream, sample_rate = runtime.voice_speak_stream("**你好**")
                try:
                    chunks = list(stream)
                finally:
                    stream.close()
                self.assertEqual(sample_rate, 24_000)
                self.assertEqual(b"".join(chunks), b"\x00\x00" * 240)
            finally:
                runtime.close()

        payload = json.loads(records[0]["body"])
        self.assertEqual(payload["input"], "你好")
        self.assertTrue(payload["stream"])

    def test_voice_speak_stream_surfaces_upstream_errors(self):
        base, _, server, thread = _serve(
            lambda record: (500, "application/json", b'{"error": "boom"}')
        )
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        from cat_assistant.debug_ui import DebugRuntime

        with tempfile.TemporaryDirectory() as tmp:
            voice_config = Path(tmp) / "voice-config.json"
            voice_config.write_text(
                json.dumps(
                    {
                        "wake": {"provider": "none"},
                        "tts": {"base_url": base, "streaming": True},
                    }
                ),
                encoding="utf-8",
            )
            runtime = DebugRuntime(
                Path(tmp) / "events.jsonl",
                voice_config_path=voice_config,
            )
            try:
                stream, _ = runtime.voice_speak_stream("你好")
                with self.assertRaises(VoiceServiceError):
                    next(stream)
                stream.close()
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
