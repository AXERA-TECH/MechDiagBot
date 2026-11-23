"""Local dependency-free debug UI for the Cat assistant."""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from cat_assistant.adapters.runtime_config import RuntimeConfigStore
from cat_assistant.adapters.telemetry import fault_code_catalog
from cat_assistant.application.loop import AgentLoop
from cat_assistant.bootstrap import build_demo_app, load_env_file
from cat_assistant.adapters.simulation import get_simulation_scenario, simulation_catalog
from cat_assistant.domain.models import MemoryRecord, Utterance
from cat_assistant.voice.config import load_voice_config


UI_ROOT = Path(__file__).with_name("ui")
_VOICE_STREAM_END = object()


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class DebugRuntime:
    def __init__(
        self,
        event_path: Path,
        config_path: Path | None = None,
        *,
        voice_config_path: Path | None = None,
        simulation_scenario: str | None = None,
        simulation_phase: int = 0,
        live_fleet: bool = False,
    ) -> None:
        self.event_path = event_path
        self.config_path = config_path or event_path.with_name("runtime-config.json")
        self.voice_config_path = (
            voice_config_path or event_path.with_name("voice-config.json")
        )
        self.simulation_scenario = simulation_scenario
        self.simulation_phase = simulation_phase
        self.live_fleet = live_fleet
        self.config_store = RuntimeConfigStore(self.config_path)
        self.app = build_demo_app(
            event_path,
            config_path=self.config_path,
            simulation_scenario=simulation_scenario,
            simulation_phase=simulation_phase,
            live_fleet=live_fleet,
        )
        self._loop = asyncio.new_event_loop()
        self._call_lock = threading.Lock()
        self._voice_service = None
        self._voice_lock = threading.Lock()

    def call(self, make_coroutine: Callable[[], Any]) -> Any:
        # ThreadingHTTPServer may invoke handlers on different threads. Run all
        # async application work on one shared loop, serialised by this lock, so
        # session locks and store locks never cross event loops. The coroutine
        # is built inside the lock so every call observes the current ``self.app``
        # even across a hot model-config reload that swaps it.
        with self._call_lock:
            return self._loop.run_until_complete(make_coroutine())

    def close(self) -> None:
        try:
            self.call(lambda: self.app.shutdown())
        finally:
            self._loop.close()

    def events(self, limit: int = 120) -> list[dict[str, Any]]:
        if not self.event_path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            with self.event_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            records.append(value)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return records[-max(1, min(limit, 500)) :]

    def state(
        self,
        machine_id: str,
        session_id: str,
        operator_id: str = "debug-operator",
    ) -> dict[str, Any]:
        # Empty machine_id is the general, unbound lane: no telemetry read, only
        # device-independent capabilities/configuration plus that lane's turns.
        machine_id = machine_id.strip()
        machine = (
            self.call(lambda: self.app.get_machine_snapshot(machine_id))
            if machine_id
            else None
        )
        turns = self.call(
            lambda: self.app.get_recent_turns(
                session_id,
                machine_id=machine_id,
                operator_id=operator_id,
                limit=30,
            )
        )
        return self._state_payload(machine, turns, session_id)

    def _state_payload(
        self,
        machine: Any,
        turns: tuple[Any, ...],
        session_id: str,
    ) -> dict[str, Any]:
        manifests = self.app.plugin_manager.plugins if self.app.plugin_manager else ()
        simulation = None
        if self.simulation_scenario:
            scenario = get_simulation_scenario(self.simulation_scenario)
            sample = scenario.sample(self.simulation_phase)
            simulation = {
                "scenario_id": scenario.scenario_id,
                "title": scenario.title,
                "description": scenario.description,
                "phase": self.simulation_phase,
                "phase_count": len(scenario.samples),
                "sample": _jsonable(sample),
            }
        return {
            "machine": _jsonable(machine) if machine is not None else None,
            "tools": _jsonable(self.app.tool_specs),
            "capabilities": _jsonable(self.app.capability_inventory),
            "fault_catalog": fault_code_catalog(),
            "plugins": _jsonable(manifests),
            "mcp_servers": self.config_store.snapshot()["mcp_servers"],
            "skills": [],
            "configuration": self.configuration(),
            "session_id": session_id,
            "events": self.events(),
            "turns": _jsonable(turns),
            "simulation": simulation,
        }

    def simulations(self) -> list[dict[str, Any]]:
        return [
            {
                "scenario_id": scenario.scenario_id,
                "title": scenario.title,
                "description": scenario.description,
                "phase_count": len(scenario.samples),
                "phases": [
                    {"phase": index, "name": sample.phase, "operator_note": sample.operator_note}
                    for index, sample in enumerate(scenario.samples)
                ],
            }
            for scenario in simulation_catalog().values()
        ]

    def machines(self) -> list[dict[str, Any]]:
        snapshots = self.call(lambda: self.app.list_machines())
        return [_jsonable(snapshot) for snapshot in snapshots]

    def turn(
        self,
        *,
        text: str,
        session_id: str,
        machine_id: str,
        operator_id: str,
    ) -> dict[str, Any]:
        response = self.call(
            lambda: self.app.handle(
                Utterance(
                    text=text,
                    session_id=session_id,
                    machine_id=machine_id,
                    operator_id=operator_id,
                )
            )
        )
        return {
            "response": _jsonable(response),
            "state": self.state(machine_id, session_id, operator_id),
        }

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete a session's backend history so the UI can stay in sync."""
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id is required")
        removed = self.call(lambda: self.app.clear_session(session_id))
        return {"ok": True, "session_id": session_id, "removed_turns": removed}

    # ---------------------------------------------------------------- 语音
    def _voice(self):
        with self._voice_lock:
            if self._voice_service is None:
                try:
                    from cat_assistant.voice.web import WebVoiceService
                except ImportError as exc:
                    raise RuntimeError(
                        "未安装语音依赖：pip install 'equip-ai-agent[voice]'"
                    ) from exc
                self._voice_service = WebVoiceService(
                    load_voice_config(self.voice_config_path)
                )
            return self._voice_service

    def voice_status(self) -> dict[str, Any]:
        try:
            from cat_assistant.voice.web import voice_status
        except ImportError as exc:
            return {"available": False, "error": str(exc)}
        try:
            return voice_status(load_voice_config(self.voice_config_path))
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def voice_transcribe(self, pcm: bytes, sample_rate: int) -> str:
        service = self._voice()
        return asyncio.run(service.transcribe(pcm, sample_rate))

    def voice_speak(self, text: str) -> bytes:
        service = self._voice()
        return asyncio.run(service.speak_wav(text))

    def voice_speak_stream(self, text: str) -> tuple[Iterator[bytes], int]:
        """返回流式 PCM 与播放采样率；首块到达即可开始播放。"""

        service = self._voice()
        chunks: queue.Queue[Any] = queue.Queue()
        cancelled = threading.Event()

        async def produce() -> None:
            try:
                async for chunk in service.stream_pcm(text):
                    if cancelled.is_set():
                        break
                    chunks.put(chunk)
            except BaseException as exc:  # noqa: BLE001 - 转交给 HTTP 线程
                chunks.put(exc)
            finally:
                chunks.put(_VOICE_STREAM_END)

        def worker() -> None:
            try:
                asyncio.run(produce())
            except BaseException as exc:  # noqa: BLE001
                chunks.put(exc)
                chunks.put(_VOICE_STREAM_END)

        def iterate() -> Iterator[bytes]:
            try:
                while True:
                    item = chunks.get()
                    if item is _VOICE_STREAM_END:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                # 客户端断开时停止继续合成，并让异步生成器尽快关闭上游连接。
                cancelled.set()

        threading.Thread(
            target=worker, name="cat-tts-stream", daemon=True
        ).start()
        return iterate(), service.config.audio.output_sample_rate

    def remember(self, payload: dict[str, Any]) -> dict[str, Any]:
        memory = MemoryRecord(
            content=str(payload.get("content", "")).strip(),
            kind=str(payload.get("kind", "fact")),
            source="debug-ui",
            machine_id=payload.get("machine_id") or None,
            operator_id=payload.get("operator_id") or None,
            confidence=float(payload.get("confidence", 1.0)),
        )
        self.call(lambda: self.app.remember(memory))
        return {"ok": True, "memory": _jsonable(memory)}

    def configuration(self) -> dict[str, Any]:
        config = self.config_store.snapshot()
        adapter_name = self.app.model_adapter_name
        config["runtime"] = {
            "model_adapter": adapter_name,
            "model_config_applied": (
                config["model"]["provider"] == "demo"
                or adapter_name == "OpenAICompatibleModel"
            ),
            "mcp_host_available": False,
            "plugin_hot_load_available": False,
        }
        return config

    def save_model_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Save into the active profile (the v1 single-model contract)."""
        return self.save_model_profile(payload, activate=True)

    def save_model_profile(
        self, payload: dict[str, Any], *, activate: bool | None = None
    ) -> dict[str, Any]:
        """Create or update one model profile.

        Only a change to the profile the runtime is *using* needs the app
        rebuilt, so editing an inactive profile is a plain write. ``activate``
        forces the saved profile live; by default it goes live only if it
        already was."""
        snapshot = self.config_store.snapshot()
        wanted = str(payload.get("id") or "").strip()
        if activate is None:
            activate = bool(wanted) and wanted == snapshot["active_model_id"]
        model = self.config_store.upsert_model(payload)
        if not activate:
            return {"ok": True, "model": model, "configuration": self.configuration()}
        return self._apply_model(
            lambda: self.config_store.activate_model(model["id"]),
            snapshot,
            model=model,
        )

    def activate_model_profile(self, item_id: str) -> dict[str, Any]:
        snapshot = self.config_store.snapshot()
        return self._apply_model(
            lambda: self.config_store.activate_model(str(item_id)), snapshot
        )

    def delete_model_profile(self, item_id: str) -> dict[str, Any]:
        snapshot = self.config_store.snapshot()
        active = str(item_id) == snapshot["active_model_id"]
        result = self.config_store.delete_model(str(item_id))
        if not active:
            return {"ok": True, "model": result, "configuration": self.configuration()}
        # Deleting the live profile promotes another one, so the app has to
        # follow it over.
        return self._apply_model(lambda: result, snapshot)

    def _apply_model(
        self,
        mutate: Callable[[], dict[str, Any]],
        snapshot: dict[str, Any],
        *,
        model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Apply the new model without a process restart: persist, then rebuild the
        # app from the persisted config and swap it in. If the rebuild/startup
        # fails (e.g. an invalid provider config), restore the last good config
        # and keep the current app serving, so a bad save neither takes the
        # console down nor leaves an unbuildable file for the next restart.
        applied = mutate()
        try:
            self._reload_app()
        except Exception:
            self.config_store.restore_models(
                snapshot["models"], snapshot["active_model_id"]
            )
            raise
        return {
            "ok": True,
            "model": model or applied,
            "model_adapter": self.app.model_adapter_name,
            "configuration": self.configuration(),
        }

    def _reload_app(self) -> None:
        # Build the replacement outside the loop lock (pure construction, no
        # network) so in-flight turns are not blocked while it is assembled.
        new_app = build_demo_app(
            self.event_path,
            config_path=self.config_path,
            simulation_scenario=self.simulation_scenario,
            simulation_phase=self.simulation_phase,
            live_fleet=self.live_fleet,
        )
        self.call(lambda: self._swap_app(new_app))

    async def _swap_app(self, new_app: AgentLoop) -> None:
        # Start the replacement before touching the live reference; if startup
        # fails, tear it back down and keep the previous app in place. Runs under
        # the shared loop lock, so no turn straddles the swap.
        try:
            await new_app.startup()
        except BaseException:
            await new_app.shutdown()
            raise
        previous = self.app
        self.app = new_app
        await previous.shutdown()

    def save_mcp_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "mcp": self.config_store.upsert_mcp(payload)}

    def delete_mcp_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.config_store.delete_mcp(str(payload.get("id", "")))
        return {"ok": True}

    def save_plugin_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "plugin": self.config_store.upsert_plugin(payload)}

    def delete_plugin_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.config_store.delete_plugin(str(payload.get("id", "")))
        return {"ok": True}


class DebugHandler(BaseHTTPRequestHandler):
    runtime: DebugRuntime

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the terminal focused on useful application logs.
        return

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _query(self) -> dict[str, str]:
        parsed = parse_qs(urlparse(self.path).query)
        return {key: values[-1] for key, values in parsed.items() if values}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True})
            return
        if parsed.path == "/api/state":
            query = self._query()
            machine_id = query.get("machine_id", "")
            session_id = query.get("session_id", "debug-session")
            operator_id = query.get("operator_id", "debug-operator")
            try:
                self._json(
                    HTTPStatus.OK,
                    self.runtime.state(machine_id, session_id, operator_id),
                )
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, type(exc).__name__)
            return
        if parsed.path == "/api/events":
            self._json(HTTPStatus.OK, {"events": self.runtime.events()})
            return
        if parsed.path == "/api/simulations":
            self._json(HTTPStatus.OK, {"simulations": self.runtime.simulations()})
            return
        if parsed.path == "/api/machines":
            self._json(HTTPStatus.OK, {"machines": self.runtime.machines()})
            return
        if parsed.path == "/api/config":
            self._json(HTTPStatus.OK, self.runtime.configuration())
            return
        if parsed.path == "/api/voice/status":
            self._json(HTTPStatus.OK, self.runtime.voice_status())
            return

        relative = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
        target = (UI_ROOT / relative).resolve()
        if not target.is_file() or not target.is_relative_to(UI_ROOT.resolve()):
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".woff2": "font/woff2",
            ".woff": "font/woff",
        }.get(target.suffix, "application/octet-stream")
        try:
            self._send(HTTPStatus.OK, target.read_bytes(), content_type)
        except OSError:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length < 0 or length > 1_000_000:
            raise ValueError("request body is too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _read_body(self, limit: int) -> bytes:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length <= 0 or length > limit:
            raise ValueError("录音数据为空或过大")
        return self.rfile.read(length)

    def _handle_voice(self, parsed) -> None:
        """网页按键录音：整段转写与整段合成，不走本地唤醒/VAD。"""

        if parsed.path == "/api/voice/transcribe":
            query = self._query()
            raw_rate = (
                query.get("sample_rate")
                or self.headers.get("X-Sample-Rate")
                or "16000"
            )
            try:
                sample_rate = int(raw_rate)
            except ValueError as exc:
                raise ValueError("sample_rate 无效") from exc
            pcm = self._read_body(8_000_000)
            try:
                text = self.runtime.voice_transcribe(pcm, sample_rate)
            except RuntimeError as exc:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            self._json(HTTPStatus.OK, {"text": text})
            return
        if parsed.path in {"/api/voice/speak", "/api/voice/speak/stream"}:
            payload = self._read_json()
            text = str(payload.get("text", "")).strip()
            if not text or len(text) > 4_000:
                raise ValueError("text 必须为 1-4000 个字符")
            if parsed.path.endswith("/stream"):
                self._stream_voice_pcm(text)
                return
            try:
                wav = self.runtime.voice_speak(text)
            except RuntimeError as exc:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            if not wav:
                raise ValueError("TTS 返回空音频")
            self._send(HTTPStatus.OK, wav, "audio/wav")
            return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def _stream_voice_pcm(self, text: str) -> None:
        """以裸 PCM 流式返回 TTS；HTTP/1.0 用连接关闭作为流结束。"""

        stream, sample_rate = self.runtime.voice_speak_stream(text)
        try:
            first = next(stream)
        except StopIteration as exc:
            stream.close()
            raise ValueError("TTS 返回空音频") from exc
        except RuntimeError as exc:
            stream.close()
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return

        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/pcm")
        self.send_header("X-Sample-Rate", str(sample_rate))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(first)
            self.wfile.flush()
            for chunk in stream:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            # 响应头已经发出，不能在同一个连接上再写 JSON 错误；直接断开，
            # 浏览器会停止播放已收到的音频。
            pass
        finally:
            stream.close()


    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/voice/"):
                self._handle_voice(parsed)
                return
            payload = self._read_json()
            if self.path == "/api/turn":
                text = str(payload.get("text", "")).strip()
                if not text or len(text) > 2_000:
                    raise ValueError("text must contain 1-2000 characters")
                result = self.runtime.turn(
                    text=text,
                    session_id=str(payload.get("session_id", "debug-session")),
                    machine_id=str(payload.get("machine_id", "")),
                    operator_id=str(payload.get("operator_id", "debug-operator")),
                )
                self._json(HTTPStatus.OK, result)
                return
            if self.path == "/api/remember":
                self._json(HTTPStatus.OK, self.runtime.remember(payload))
                return
            if self.path == "/api/session/delete":
                self._json(
                    HTTPStatus.OK,
                    self.runtime.delete_session(str(payload.get("session_id", ""))),
                )
                return
            if self.path == "/api/config/model":
                self._json(HTTPStatus.OK, self.runtime.save_model_config(payload))
                return
            if self.path == "/api/config/model/save":
                self._json(HTTPStatus.OK, self.runtime.save_model_profile(payload))
                return
            if self.path == "/api/config/model/activate":
                self._json(
                    HTTPStatus.OK,
                    self.runtime.activate_model_profile(str(payload.get("id", ""))),
                )
                return
            if self.path == "/api/config/model/delete":
                self._json(
                    HTTPStatus.OK,
                    self.runtime.delete_model_profile(str(payload.get("id", ""))),
                )
                return
            if self.path == "/api/config/mcp":
                self._json(HTTPStatus.OK, self.runtime.save_mcp_config(payload))
                return
            if self.path == "/api/config/mcp/delete":
                self._json(HTTPStatus.OK, self.runtime.delete_mcp_config(payload))
                return
            if self.path == "/api/config/plugin":
                self._json(HTTPStatus.OK, self.runtime.save_plugin_config(payload))
                return
            if self.path == "/api/config/plugin/delete":
                self._json(HTTPStatus.OK, self.runtime.delete_plugin_config(payload))
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, type(exc).__name__)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    event_path: Path | None = None,
    config_path: Path | None = None,
    simulation_scenario: str | None = None,
    simulation_phase: int = 0,
    voice_config_path: Path | None = None,
):
    runtime = DebugRuntime(
        event_path or Path("runtime/debug-events.jsonl"),
        config_path=config_path,
        voice_config_path=voice_config_path,
        simulation_scenario=simulation_scenario,
        simulation_phase=simulation_phase,
        # A live server shows the fleet "breathing": faults reshuffle every 10
        # minutes. Scenario mode ignores this (it replays a scripted timeline).
        live_fleet=True,
    )

    class Handler(DebugHandler):
        pass

    Handler.runtime = runtime
    server = ThreadingHTTPServer((host, port), Handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    return server


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Run the Cat Assistant local debug UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--event-path", type=Path, default=Path("runtime/debug-events.jsonl"))
    parser.add_argument("--config-path", type=Path, default=Path("runtime/runtime-config.json"))
    parser.add_argument(
        "--voice-config-path",
        type=Path,
        default=Path("runtime/voice-config.json"),
        help="语音 ASR/TTS 配置（网页录音使用）",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(simulation_catalog()),
        help="run a deterministic multi-stage equipment simulation",
    )
    parser.add_argument("--phase", type=int, default=0, help="initial simulation phase")
    args = parser.parse_args()
    server = create_server(
        args.host, args.port, args.event_path, args.config_path,
        simulation_scenario=args.scenario, simulation_phase=args.phase,
        voice_config_path=args.voice_config_path,
    )
    print(f"Cat Assistant debug UI: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server.runtime.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
