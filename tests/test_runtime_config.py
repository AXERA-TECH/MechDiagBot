from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cat_assistant.adapters.runtime_config import RuntimeConfigStore
from cat_assistant.bootstrap import build_demo_app


class RuntimeConfigStoreTests(unittest.TestCase):
    def test_configuration_is_persistent_and_secrets_are_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = RuntimeConfigStore(path)
            store.save_model(
                {
                    "provider": "openai_compatible",
                    "model": "edge-model",
                    "base_url": "http://localhost:8000/v1",
                    "api_key_env": "EDGE_MODEL_API_KEY",
                    "temperature": 0.1,
                    "max_tokens": 512,
                    "timeout_seconds": 20,
                    "max_steps": 3,
                }
            )
            item = store.upsert_mcp(
                {
                    "name": "manuals",
                    "transport": "stdio",
                    "command": "mcp-manuals",
                    "env_keys": ["MANUALS_TOKEN"],
                    "tool_allowlist": ["search_manual"],
                }
            )

            reloaded = RuntimeConfigStore(path).snapshot()
            self.assertEqual(reloaded["model"]["api_key_env"], "EDGE_MODEL_API_KEY")
            self.assertEqual(reloaded["model"]["planning_mode"], "auto")
            self.assertEqual(reloaded["model"]["model_call_timeout_seconds"], 60.0)
            self.assertEqual(reloaded["mcp_servers"][0]["id"], item["id"])
            self.assertNotIn("super-secret-value", path.read_text(encoding="utf-8"))

    def test_axllm_provider_persists_edge_model_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            RuntimeConfigStore(path).save_model(
                {
                    "provider": "axllm",
                    "model": "Qwen3-4B",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "timeout_seconds": 60,
                    # A weak edge model can't reliably one-shot a tool-call DAG, so
                    # the recommended deployment plans deterministically (rule) and
                    # only uses the model to write the diagnosis; /no_think is on and
                    # ChatML stop tokens bound a server that forgets its EOS.
                    "planning_mode": "rule",
                    "disable_thinking": True,
                    "stop": ["<|im_end|>", "<|im_start|>"],
                }
            )
            reloaded = RuntimeConfigStore(path).snapshot()["model"]
            self.assertEqual(reloaded["provider"], "axllm")
            self.assertEqual(reloaded["planning_mode"], "rule")
            self.assertTrue(reloaded["disable_thinking"])
            self.assertEqual(reloaded["stop"], ["<|im_end|>", "<|im_start|>"])

    def test_invalid_plugin_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RuntimeConfigStore(Path(directory) / "config.json")
            with self.assertRaisesRegex(ValueError, "reference"):
                store.upsert_plugin(
                    {"name": "unsafe", "reference": "missing_factory", "config": {}}
                )

    def test_each_model_keeps_its_own_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = RuntimeConfigStore(path)
            edge = store.upsert_model(
                {
                    "name": "边缘小模型",
                    "provider": "axllm",
                    "model": "Qwen3-1.7B",
                    "base_url": "http://127.0.0.1:8019/v1",
                    "temperature": 0.1,
                    "max_tokens": 512,
                    "timeout_seconds": 600,
                    "max_steps": 5,
                    "planning_mode": "rule",
                    "turn_timeout_seconds": 1200,
                }
            )
            cloud = store.upsert_model(
                {
                    "name": "云端大模型",
                    "provider": "vllm",
                    "model": "Qwen/Qwen3-8B",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "timeout_seconds": 60,
                    "max_steps": 6,
                    "planning_mode": "tool_call",
                    "turn_timeout_seconds": 300,
                }
            )
            self.assertNotEqual(edge["id"], cloud["id"])

            # Activating a profile switches the whole parameter set, not just the
            # model name -- that is the point of per-model configuration.
            store.activate_model(cloud["id"])
            active = RuntimeConfigStore(path).snapshot()["model"]
            self.assertEqual(active["model"], "Qwen/Qwen3-8B")
            self.assertEqual(active["temperature"], 0.7)
            self.assertEqual(active["max_steps"], 6)
            self.assertEqual(active["turn_timeout_seconds"], 300.0)

            store.activate_model(edge["id"])
            active = RuntimeConfigStore(path).snapshot()["model"]
            self.assertEqual(active["model"], "Qwen3-1.7B")
            self.assertEqual(active["temperature"], 0.1)
            self.assertEqual(active["max_steps"], 5)
            self.assertEqual(active["turn_timeout_seconds"], 1200.0)
            self.assertEqual(active["planning_mode"], "rule")

    def test_v1_config_migrates_into_a_named_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model": {
                            "provider": "axllm",
                            "model": "AXERA-TECH/Qwen3-1.7B",
                            "base_url": "http://10.0.0.1:8019/v1",
                            "api_key_env": "EDGE_MODEL_API_KEY",
                            "planning_mode": "tool_call",
                            "disable_thinking": True,
                            "stop": ["<|im_end|>"],
                            "turn_timeout_seconds": 1200.0,
                        },
                        "mcp_servers": [],
                        "plugins": [],
                    }
                ),
                encoding="utf-8",
            )
            snapshot = RuntimeConfigStore(path).snapshot()
            self.assertEqual(len(snapshot["models"]), 1)
            profile = snapshot["models"][0]
            # The migrated profile is named after the model it points at, not
            # after the built-in demo default.
            self.assertEqual(profile["name"], "AXERA-TECH/Qwen3-1.7B")
            self.assertNotEqual(profile["id"], "demo")
            self.assertEqual(profile["provider"], "axllm")
            self.assertEqual(profile["api_key_env"], "EDGE_MODEL_API_KEY")
            self.assertEqual(profile["stop"], ["<|im_end|>"])
            self.assertEqual(profile["turn_timeout_seconds"], 1200.0)
            self.assertEqual(snapshot["active_model_id"], profile["id"])
            # ``model`` is derived, so the next write drops it and leaves only
            # the profile list behind.
            store = RuntimeConfigStore(path)
            store.activate_model(profile["id"])
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("model", on_disk)
            self.assertEqual(on_disk["version"], 2)
            self.assertEqual(len(on_disk["models"]), 1)

    def test_console_save_keeps_fields_the_form_never_sends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = RuntimeConfigStore(path)
            saved = store.upsert_model(
                {
                    "provider": "axllm",
                    "model": "Qwen3-4B",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key_env": "EDGE_MODEL_API_KEY",
                    "stop": ["<|im_end|>", "<|im_start|>"],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "timeout_seconds": 60,
                    "max_steps": 4,
                }
            )
            store.activate_model(saved["id"])
            # The console form has no stop/api_key_env inputs. Saving from it must
            # not wipe what is on disk.
            store.upsert_model(
                {
                    "id": saved["id"],
                    "provider": "axllm",
                    "model": "Qwen3-4B",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key": "",
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "timeout_seconds": 60,
                    "max_steps": 4,
                }
            )
            reloaded = RuntimeConfigStore(path).snapshot()["model"]
            self.assertEqual(reloaded["id"], saved["id"])
            self.assertEqual(reloaded["stop"], ["<|im_end|>", "<|im_start|>"])
            self.assertEqual(reloaded["api_key_env"], "EDGE_MODEL_API_KEY")
            self.assertEqual(reloaded["temperature"], 0.3)
            self.assertEqual(reloaded["max_tokens"], 2048)

    def test_last_model_profile_cannot_be_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RuntimeConfigStore(Path(directory) / "config.json")
            extra = store.upsert_model(
                {
                    "provider": "vllm",
                    "model": "Qwen/Qwen3-4B",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "temperature": 0.1,
                    "max_tokens": 512,
                    "timeout_seconds": 30,
                    "max_steps": 4,
                }
            )
            store.activate_model(extra["id"])
            # Deleting the live profile promotes a survivor.
            store.delete_model(extra["id"])
            self.assertEqual(store.snapshot()["active_model_id"], "demo")
            # The runtime always needs something to boot with.
            with self.assertRaisesRegex(ValueError, "at least one"):
                store.delete_model("demo")

    def test_invalid_planning_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RuntimeConfigStore(Path(directory) / "config.json")
            model = store.snapshot()["model"]
            model["planning_mode"] = "semantic_rag"
            with self.assertRaisesRegex(ValueError, "planning_mode"):
                store.save_model(model)

    def test_cli_composition_root_selects_configured_model_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            RuntimeConfigStore(config_path).save_model(
                {
                    "provider": "vllm",
                    "model": "Qwen/Qwen3-4B",
                    "base_url": "http://localhost:8000/v1",
                    "api_key_env": "",
                    "temperature": 0.1,
                    "max_tokens": 256,
                    "timeout_seconds": 10,
                    "max_steps": 2,
                }
            )
            app = build_demo_app(root / "events.jsonl", config_path=config_path)
            self.assertEqual(app.model_adapter_name, "OpenAICompatibleModel")


if __name__ == "__main__":
    unittest.main()
