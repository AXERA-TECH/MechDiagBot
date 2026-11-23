from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cat_assistant.debug_ui import DebugRuntime, UI_ROOT


class DebugUiTests(unittest.TestCase):
    def test_runtime_exposes_machine_extensions_and_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = DebugRuntime(Path(directory) / "events.jsonl")
            try:
                state = runtime.state("cat-306-demo", "ui-session")
                self.assertEqual(state["machine"]["model"], "Cat 306 CR")
                self.assertEqual(
                    {tool["name"] for tool in state["tools"]},
                    {"get_machine_status", "search_manual"},
                )
                self.assertTrue(state["capabilities"])
                self.assertTrue(all(item["enabled"] for item in state["capabilities"]))
                self.assertEqual(state["capabilities"][0]["owner"], "cat.core")
                self.assertEqual(state["configuration"]["model"]["provider"], "demo")
                self.assertTrue(
                    state["configuration"]["runtime"]["model_config_applied"]
                )

                result = runtime.turn(
                    text="下一次保养是什么时候？",
                    session_id="ui-session",
                    machine_id="cat-306-demo",
                    operator_id="debug-operator",
                )
                self.assertEqual(result["response"]["category"], "information")
                self.assertTrue(result["state"]["events"])
                restored = runtime.state(
                    "cat-306-demo",
                    "ui-session",
                    "debug-operator",
                )
                self.assertEqual(len(restored["turns"]), 1)
                self.assertEqual(
                    restored["turns"][0]["user_text"],
                    "下一次保养是什么时候？",
                )

                runtime.save_model_config(
                    {
                        "provider": "vllm",
                        "model": "Qwen/Qwen3-4B",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "api_key_env": "CAT_MODEL_API_KEY",
                        "temperature": 0.2,
                        "max_tokens": 1024,
                        "timeout_seconds": 30,
                        "max_steps": 4,
                    }
                )
                runtime.save_mcp_config(
                    {
                        "name": "设备知识库",
                        "transport": "streamable_http",
                        "endpoint": "http://127.0.0.1:9000/mcp",
                        "tool_allowlist": ["search_manual"],
                        "enabled": True,
                    }
                )
                runtime.save_plugin_config(
                    {
                        "name": "diagnostic-extension",
                        "reference": "example.plugin:create_plugin",
                        "config": {"read_only": True},
                        "enabled": False,
                    }
                )
                configured = runtime.configuration()
                self.assertEqual(configured["model"]["provider"], "vllm")
                # Saving now hot-reloads the running app, so the live adapter
                # switches immediately instead of waiting for a restart.
                self.assertTrue(configured["runtime"]["model_config_applied"])
                self.assertEqual(
                    configured["runtime"]["model_adapter"], "OpenAICompatibleModel"
                )
                self.assertEqual(configured["mcp_servers"][0]["status"], "configured")
                self.assertEqual(configured["plugins"][0]["status"], "configured")
            finally:
                runtime.close()

    def test_saving_model_config_hot_reloads_without_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = DebugRuntime(Path(directory) / "events.jsonl")
            try:
                # The console boots on the offline demo model.
                self.assertEqual(runtime.app.model_adapter_name, "DemoRuleBasedModel")

                result = runtime.save_model_config(
                    {
                        "provider": "openai_compatible",
                        "model": "AXERA-TECH/Qwen3.5-2B",
                        "base_url": "http://127.0.0.1:8019/v1",
                        "temperature": 0.1,
                        "max_tokens": 1024,
                        "timeout_seconds": 30,
                        "max_steps": 4,
                    }
                )

                # The live app is swapped in place — no process restart, and the
                # applied-flag/adapter both reflect the new model right away.
                self.assertEqual(result["model_adapter"], "OpenAICompatibleModel")
                self.assertEqual(
                    runtime.app.model_adapter_name, "OpenAICompatibleModel"
                )
                self.assertTrue(
                    runtime.configuration()["runtime"]["model_config_applied"]
                )
            finally:
                runtime.close()

    def test_editing_an_inactive_profile_leaves_the_running_model_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = DebugRuntime(Path(directory) / "events.jsonl")
            try:
                self.assertEqual(runtime.app.model_adapter_name, "DemoRuleBasedModel")

                # A second profile is just a write: the console keeps running on
                # whatever is active until you say otherwise.
                saved = runtime.save_model_profile(
                    {
                        "name": "边缘小模型",
                        "provider": "openai_compatible",
                        "model": "AXERA-TECH/Qwen3-1.7B",
                        "base_url": "http://127.0.0.1:8019/v1",
                        "temperature": 0.1,
                        "max_tokens": 512,
                        "timeout_seconds": 600,
                        "max_steps": 5,
                        "planning_mode": "tool_call",
                        "turn_timeout_seconds": 1200,
                    }
                )["model"]
                self.assertEqual(runtime.app.model_adapter_name, "DemoRuleBasedModel")
                configuration = runtime.configuration()
                self.assertEqual(len(configuration["models"]), 2)
                self.assertEqual(configuration["model"]["provider"], "demo")

                # Activating swaps the adapter and the whole parameter set.
                result = runtime.activate_model_profile(saved["id"])
                self.assertEqual(result["model_adapter"], "OpenAICompatibleModel")
                active = runtime.configuration()["model"]
                self.assertEqual(active["id"], saved["id"])
                self.assertEqual(active["max_tokens"], 512)
                self.assertEqual(active["max_steps"], 5)
                self.assertEqual(active["turn_timeout_seconds"], 1200.0)

                # Deleting the live profile promotes a survivor and the app
                # follows it back to the demo adapter.
                runtime.delete_model_profile(saved["id"])
                self.assertEqual(runtime.app.model_adapter_name, "DemoRuleBasedModel")
                self.assertEqual(runtime.configuration()["active_model_id"], "demo")
            finally:
                runtime.close()

    def test_failed_model_activation_rolls_back_and_keeps_serving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = DebugRuntime(Path(directory) / "events.jsonl")
            try:
                # openai_compatible has no default endpoint, so an empty base_url
                # only blows up when the app is rebuilt -- exactly the case the
                # rollback exists for.
                broken = runtime.save_model_profile(
                    {
                        "provider": "openai_compatible",
                        "model": "no-endpoint",
                        "base_url": "",
                        "temperature": 0.1,
                        "max_tokens": 512,
                        "timeout_seconds": 30,
                        "max_steps": 4,
                    }
                )["model"]
                before = runtime.configuration()

                with self.assertRaises(ValueError):
                    runtime.activate_model_profile(broken["id"])

                after = runtime.configuration()
                self.assertEqual(runtime.app.model_adapter_name, "DemoRuleBasedModel")
                self.assertEqual(after["active_model_id"], before["active_model_id"])
                self.assertEqual(after["models"], before["models"])
            finally:
                runtime.close()

    def test_delete_session_syncs_backend_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = DebugRuntime(Path(directory) / "events.jsonl")
            try:
                runtime.turn(
                    text="下一次保养是什么时候？",
                    session_id="ui-session",
                    machine_id="cat-306-demo",
                    operator_id="debug-operator",
                )
                before = runtime.state("cat-306-demo", "ui-session", "debug-operator")
                self.assertEqual(len(before["turns"]), 1)

                result = runtime.delete_session("ui-session")
                self.assertEqual(
                    result,
                    {"ok": True, "session_id": "ui-session", "removed_turns": 1},
                )

                after = runtime.state("cat-306-demo", "ui-session", "debug-operator")
                self.assertEqual(after["turns"], [])
                self.assertTrue(
                    any(
                        event["event_type"] == "session/cleared"
                        for event in runtime.events()
                    )
                )
                with self.assertRaises(ValueError):
                    runtime.delete_session("   ")
            finally:
                runtime.close()

    def test_general_chat_before_device_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = DebugRuntime(Path(directory) / "events.jsonl")
            try:
                result = runtime.turn(
                    text="你好",
                    session_id="general-ui",
                    machine_id="",
                    operator_id="debug-operator",
                )
                self.assertIsNone(result["state"]["machine"])
                self.assertTrue(result["response"]["text"])
                self.assertTrue(result["state"]["capabilities"])

                restored = runtime.state("", "general-ui", "debug-operator")
                self.assertIsNone(restored["machine"])
                self.assertEqual(len(restored["turns"]), 1)
                self.assertEqual(restored["turns"][0]["machine_id"], "")
                self.assertEqual(restored["turns"][0]["user_text"], "你好")
            finally:
                runtime.close()

    def test_static_assets_are_packaged(self) -> None:
        for name in ("index.html", "style.css", "app.js"):
            self.assertTrue((UI_ROOT / name).is_file())

    def test_static_ui_exposes_accessible_navigation_and_forms(self) -> None:
        html = (UI_ROOT / "index.html").read_text(encoding="utf-8")
        css = (UI_ROOT / "style.css").read_text(encoding="utf-8")
        script = (UI_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('class="skip-link"', html)
        self.assertIn('aria-label="工作台 · 通用助手"', html)
        self.assertIn('role="tablist"', html)
        self.assertIn('for="turn-input"', html)
        self.assertIn('aria-labelledby="mcp-dialog-title"', html)
        self.assertIn("@media (prefers-reduced-motion: reduce)", css)
        self.assertIn("input:focus-visible", css)
        self.assertIn("new Intl.DateTimeFormat", script)
        self.assertIn("<details class=\"event-row\">", script)
        self.assertIn('id="delete-session-btn"', html)
        self.assertIn("/api/session/delete", script)

        # The workbench nav entry is the unbound general agent; clicking a
        # device attaches that machine instead.
        self.assertIn("selectGeneralAgent", script)

        # Only voice-input turns speak the reply; typed turns stay silent.
        # The reply is streamed as PCM so playback starts on the first chunk.
        self.assertIn('id="voice-btn"', html)
        self.assertNotIn('id="speak-btn"', html)
        for endpoint in (
            "/api/voice/status",
            "/api/voice/speak/stream",
            "/api/voice/speak",
            "/api/voice/transcribe",
        ):
            self.assertIn(endpoint, script)
        self.assertIn("scheduleVoicePcm", script)

        # One click opens a hands-free session: local VAD auto-sends on
        # pause, the reply plays, then listening resumes until the next click.
        self.assertIn("toggleVoiceSession", script)
        self.assertIn("handleVoiceFrame", script)
        self.assertIn("waitVoicePlaybackDone", script)
        self.assertIn("resumeVoiceListening", script)
        self.assertNotIn("按住麦克风", html)
        self.assertNotIn("按住麦克风", script)

        # The model screen is a profile list + one editor, so every profile
        # carries its own connection/inference/timeout fields.
        self.assertIn('id="model-list"', html)
        self.assertIn('id="add-model-btn"', html)
        for field in ("model-temperature", "model-max-tokens", "model-max-steps",
                      "model-planning-mode", "model-disable-thinking",
                      "model-call-timeout", "tool-call-timeout", "turn-timeout"):
            self.assertIn(f'id="{field}"', html)
        for endpoint in ("/api/config/model/save", "/api/config/model/activate",
                         "/api/config/model/delete"):
            self.assertIn(endpoint, script)

    def test_runtime_exposes_simulation_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = DebugRuntime(
                Path(directory) / "events.jsonl",
                simulation_scenario="coolant-overheat",
                simulation_phase=2,
            )
            try:
                state = runtime.state("cat-306-demo", "simulation-ui")
                self.assertEqual(state["simulation"]["scenario_id"], "coolant-overheat")
                self.assertEqual(state["simulation"]["phase"], 2)
                self.assertEqual(state["machine"]["fault_codes"], ["E456"])
                catalog = runtime.simulations()
                self.assertGreaterEqual(len(catalog), 3)
                self.assertTrue(all(item["phase_count"] >= 4 for item in catalog))
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
