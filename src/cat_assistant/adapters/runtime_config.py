"""Local configuration persistence used by the dependency-free debug console.

The store intentionally persists declarations, not live runtime objects.  A saved
MCP server or plugin is therefore "configured" until the production host has
validated and loaded it.  Secrets are referenced by environment-variable name
and are never accepted as config values.

``models`` holds one profile per model, and every profile carries its own
connection, inference and timeout parameters -- switching models therefore
switches all of them at once.  ``active_model_id`` names the profile the runtime
boots with.  The single ``model`` key that v1 files used is migrated into a
profile on load and re-derived (never stored) by :meth:`snapshot`, so an old
config file keeps working while there is only ever one source of truth.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4


MODEL_PROVIDERS = frozenset({"demo", "vllm", "ollama", "openai_compatible", "axllm"})

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "id": "demo",
    "name": "Demo 离线规则模型",
    "provider": "demo",
    "model": "demo-rule-based",
    "base_url": "",
    "api_key": "",
    "api_key_env": "",
    "temperature": 0.1,
    "max_tokens": 1024,
    "timeout_seconds": 30,
    "max_steps": 4,
    "planning_mode": "auto",
    "disable_thinking": False,
    "stop": [],
    "model_call_timeout_seconds": 60.0,
    "tool_call_timeout_seconds": 30.0,
    "turn_timeout_seconds": 120.0,
}

DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "version": 2,
    "models": [deepcopy(DEFAULT_MODEL_CONFIG)],
    "active_model_id": DEFAULT_MODEL_CONFIG["id"],
    "mcp_servers": [],
    "plugins": [],
}

# Fields the console never sends.  A save that omits them keeps whatever is on
# disk instead of silently clearing it -- the axllm template's ChatML stop
# tokens and an env-var secret reference would otherwise vanish on first edit.
_PRESERVED_WHEN_ABSENT = ("api_key_env", "stop")


class RuntimeConfigStore:
    """Small atomic JSON store for local-only UI configuration."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return deepcopy(DEFAULT_RUNTIME_CONFIG)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return deepcopy(DEFAULT_RUNTIME_CONFIG)
        if not isinstance(raw, dict):
            return deepcopy(DEFAULT_RUNTIME_CONFIG)
        data = deepcopy(DEFAULT_RUNTIME_CONFIG)
        models: list[dict[str, Any]] = []
        stored = raw.get("models")
        if isinstance(stored, list):
            models = self._migrate_all(stored)
        elif isinstance(raw.get("model"), dict):
            # v1 file: one global model. Adopt it as the single profile so the
            # saved provider/secret survive the upgrade untouched.
            models = self._migrate_all([raw["model"]])
        if not models:
            models = deepcopy(DEFAULT_RUNTIME_CONFIG["models"])
        data["models"] = models
        wanted = str(raw.get("active_model_id", "")).strip()
        ids = [item["id"] for item in models]
        data["active_model_id"] = wanted if wanted in ids else ids[0]
        for key in ("mcp_servers", "plugins"):
            if isinstance(raw.get(key), list):
                data[key] = [item for item in raw[key] if isinstance(item, dict)]
        return data

    def _migrate(self, item: dict[str, Any]) -> dict[str, Any]:
        """Fold one stored (or freshly read) profile onto the full schema."""
        profile = deepcopy(DEFAULT_MODEL_CONFIG)
        # ``id``/``name`` identify the profile, so they may only come from the
        # item itself. Inheriting the default's would label a migrated v1 config
        # "Demo 离线规则模型" no matter which model it actually points at.
        profile["id"] = ""
        profile["name"] = ""
        profile.update({key: value for key, value in item.items() if value is not None})
        model_name = str(profile.get("model", "")).strip() or DEFAULT_MODEL_CONFIG["model"]
        profile["model"] = model_name
        profile["name"] = str(profile.get("name", "")).strip() or model_name
        profile["id"] = self._safe_identifier(profile.get("id"), model_name)
        if profile["provider"] not in MODEL_PROVIDERS:
            profile["provider"] = DEFAULT_MODEL_CONFIG["provider"]
        return profile

    def _migrate_all(self, items: Any) -> list[dict[str, Any]]:
        """Migrate a profile list, keeping ids unique.

        A hand-edited file can repeat an id; leaving duplicates would make every
        edit land on whichever copy came first.
        """
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            profile = self._migrate(item)
            if profile["id"] in seen:
                profile["id"] = f"{profile['id'][:50]}-{uuid4().hex[:6]}"
            seen.add(profile["id"])
            rows.append(profile)
        return rows

    def _safe_identifier(self, value: Any, seed: str) -> str:
        try:
            return self._identifier(value, "model id")
        except ValueError:
            slug = "".join(
                character if character in "abcdefghijklmnopqrstuvwxyz0123456789" else "-"
                for character in seed.lower()
            ).strip("-")
            return slug[:60] or f"model-{uuid4().hex[:8]}"

    # ---- read -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = deepcopy(self._data)
            # ``model`` is derived, never stored: the profile the runtime would
            # boot with. Kept so v1 consumers (bootstrap, callers) read the same
            # shape they always did.
            data["model"] = deepcopy(self._active_model_locked())
            return data

    def _active_model_locked(self) -> dict[str, Any]:
        rows = self._data.get("models") or []
        if not rows:
            rows = [deepcopy(DEFAULT_MODEL_CONFIG)]
            self._data["models"] = rows
        active = str(self._data.get("active_model_id", ""))
        for row in rows:
            if row.get("id") == active:
                return row
        self._data["active_model_id"] = rows[0]["id"]
        return rows[0]

    # ---- write: model profiles -----------------------------------------

    def save_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Save into the *active* profile. Kept for the v1 single-model contract."""
        with self._lock:
            active = self._active_model_locked()
            merged = dict(payload)
            merged["id"] = active["id"]
            return self.upsert_model(merged)

    def upsert_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            item_id = str(payload.get("id") or "").strip()
            existing = next(
                (row for row in self._data["models"] if row.get("id") == item_id), None
            )
            if not item_id:
                # A new profile gets a readable id derived from the model name --
                # it is what the console shows under the profile title. Suffix it
                # when taken so saving the same model twice creates two profiles
                # instead of overwriting the first one.
                taken = {row.get("id") for row in self._data["models"]}
                item_id = self._safe_identifier("", str(payload.get("model", "")))
                if item_id in taken:
                    item_id = f"{item_id[:50]}-{uuid4().hex[:6]}"
            item = self._validate_model(payload, existing, fallback_id=item_id)
            rows = self._data["models"]
            for index, current in enumerate(rows):
                if current.get("id") == item["id"]:
                    rows[index] = item
                    break
            else:
                rows.append(item)
            if self._data["active_model_id"] not in [row["id"] for row in rows]:
                self._data["active_model_id"] = item["id"]
            self._write()
            return deepcopy(item)

    def activate_model(self, item_id: str) -> dict[str, Any]:
        safe_id = self._identifier(item_id, "model id")
        with self._lock:
            if not any(row.get("id") == safe_id for row in self._data["models"]):
                raise ValueError("model profile not found")
            self._data["active_model_id"] = safe_id
            self._write()
            return deepcopy(self._active_model_locked())

    def delete_model(self, item_id: str) -> dict[str, Any]:
        safe_id = self._identifier(item_id, "model id")
        with self._lock:
            rows = self._data["models"]
            if not any(row.get("id") == safe_id for row in rows):
                raise ValueError("model profile not found")
            # The runtime always needs a model to boot with, so the last profile
            # stays put rather than leaving a config nothing can start from.
            if len(rows) <= 1:
                raise ValueError("at least one model profile must remain")
            self._data["models"] = [row for row in rows if row.get("id") != safe_id]
            if self._data["active_model_id"] == safe_id:
                self._data["active_model_id"] = self._data["models"][0]["id"]
            self._write()
            return deepcopy(self._active_model_locked())

    def restore_models(
        self, models: list[dict[str, Any]], active_model_id: str
    ) -> None:
        """Put a previously read profile list back, for rollback after a failed
        hot reload."""
        with self._lock:
            rows = self._migrate_all(models)
            self._data["models"] = rows or deepcopy(DEFAULT_RUNTIME_CONFIG["models"])
            ids = [row["id"] for row in self._data["models"]]
            self._data["active_model_id"] = (
                active_model_id if active_model_id in ids else ids[0]
            )
            self._write()

    def _validate_model(
        self,
        payload: dict[str, Any],
        existing: dict[str, Any] | None = None,
        *,
        fallback_id: str = "",
    ) -> dict[str, Any]:
        existing = existing if isinstance(existing, dict) else {}
        provider = self._required(payload, "provider", 40)
        if provider not in MODEL_PROVIDERS:
            raise ValueError("unsupported model provider")
        model = self._required(payload, "model", 200)
        item: dict[str, Any] = {
            "id": self._identifier(
                fallback_id or f"model-{uuid4().hex[:8]}", "model id"
            ),
            "name": self._optional(payload, "name", 60)
            or str(existing.get("name", "")).strip()
            or model,
            "provider": provider,
            "model": model,
            "base_url": self._optional(payload, "base_url", 500),
            "api_key": self._optional(payload, "api_key", 500),
            "temperature": self._number(payload, "temperature", 0.0, 2.0),
            "max_tokens": self._integer(payload, "max_tokens", 1, 131_072),
            "timeout_seconds": self._integer(payload, "timeout_seconds", 1, 600),
            "max_steps": self._integer(payload, "max_steps", 1, 12, default=4),
            "planning_mode": self._planning_mode(payload.get("planning_mode", "auto")),
            "disable_thinking": bool(payload.get("disable_thinking", False)),
            "model_call_timeout_seconds": self._number(
                payload, "model_call_timeout_seconds", 0.1, 600.0, default=60.0
            ),
            "tool_call_timeout_seconds": self._number(
                payload, "tool_call_timeout_seconds", 0.1, 600.0, default=30.0
            ),
            "turn_timeout_seconds": self._number(
                payload, "turn_timeout_seconds", 0.1, 1800.0, default=120.0
            ),
        }
        for key in _PRESERVED_WHEN_ABSENT:
            if key in payload:
                item[key] = (
                    self._string_list(payload.get(key), key)
                    if isinstance(DEFAULT_MODEL_CONFIG[key], list)
                    else self._optional(payload, key, 200)
                )
            else:
                item[key] = deepcopy(existing.get(key) or DEFAULT_MODEL_CONFIG[key])
        return item

    # ---- write: MCP / plugins -------------------------------------------

    def upsert_mcp(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = self._identifier(payload.get("id") or f"mcp-{uuid4()}", "MCP id")
        transport = self._required(payload, "transport", 30)
        if transport not in {"streamable_http", "sse", "stdio"}:
            raise ValueError("unsupported MCP transport")
        endpoint = self._optional(payload, "endpoint", 1000)
        command = self._optional(payload, "command", 500)
        if transport == "stdio" and not command:
            raise ValueError("stdio MCP requires command")
        if transport != "stdio" and not endpoint:
            raise ValueError("remote MCP requires endpoint")
        item = {
            "id": item_id,
            "name": self._required(payload, "name", 100),
            "transport": transport,
            "endpoint": endpoint,
            "command": command,
            "arguments": self._string_list(payload.get("arguments"), "arguments"),
            "env_keys": self._string_list(payload.get("env_keys"), "env_keys"),
            "tool_allowlist": self._string_list(
                payload.get("tool_allowlist") or ["*"], "tool_allowlist"
            ),
            "enabled": bool(payload.get("enabled", False)),
            "status": "configured",
        }
        return self._upsert("mcp_servers", item)

    def delete_mcp(self, item_id: str) -> None:
        self._delete("mcp_servers", item_id)

    def upsert_plugin(self, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = self._identifier(
            payload.get("id") or f"plugin-{uuid4()}", "plugin id"
        )
        reference = self._required(payload, "reference", 300)
        if ":" not in reference:
            raise ValueError("plugin reference must look like package.module:factory")
        config = payload.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("plugin config must be a JSON object")
        item = {
            "id": item_id,
            "name": self._required(payload, "name", 100),
            "reference": reference,
            "version": self._optional(payload, "version", 60),
            "config": config,
            "enabled": bool(payload.get("enabled", False)),
            "status": "configured",
        }
        return self._upsert("plugins", item)

    def delete_plugin(self, item_id: str) -> None:
        self._delete("plugins", item_id)

    def _upsert(self, collection: str, item: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rows = self._data[collection]
            for index, current in enumerate(rows):
                if current.get("id") == item["id"]:
                    rows[index] = item
                    break
            else:
                rows.append(item)
            self._write()
            return deepcopy(item)

    def _delete(self, collection: str, item_id: str) -> None:
        safe_id = self._identifier(item_id, "id")
        with self._lock:
            self._data[collection] = [
                row for row in self._data[collection] if row.get("id") != safe_id
            ]
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _required(payload: dict[str, Any], key: str, limit: int) -> str:
        value = str(payload.get(key, "")).strip()
        if not value or len(value) > limit:
            raise ValueError(f"{key} must contain 1-{limit} characters")
        return value

    @staticmethod
    def _optional(payload: dict[str, Any], key: str, limit: int) -> str:
        value = str(payload.get(key, "")).strip()
        if len(value) > limit:
            raise ValueError(f"{key} must contain at most {limit} characters")
        return value

    @staticmethod
    def _identifier(value: Any, label: str) -> str:
        normalized = str(value or "").strip().lower().replace(" ", "-")
        if not normalized or len(normalized) > 100:
            raise ValueError(f"{label} is invalid")
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in normalized):
            raise ValueError(f"{label} may only contain letters, numbers, dot, dash and underscore")
        return normalized

    @staticmethod
    def _number(
        payload: dict[str, Any],
        key: str,
        minimum: float,
        maximum: float,
        *,
        default: float | None = None,
    ) -> float:
        try:
            value = float(payload.get(key, default))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        return value

    @staticmethod
    def _integer(
        payload: dict[str, Any],
        key: str,
        minimum: int,
        maximum: int,
        *,
        default: int | None = None,
    ) -> int:
        try:
            value = int(payload.get(key, default))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        return value

    @staticmethod
    def _string_list(value: Any, label: str) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list) or len(value) > 200:
            raise ValueError(f"{label} must be a list")
        result = [str(item).strip() for item in value if str(item).strip()]
        if any(len(item) > 500 for item in result):
            raise ValueError(f"{label} contains an item that is too long")
        return result

    @staticmethod
    def _planning_mode(value: Any) -> str:
        mode = str(value).strip()
        if mode not in {"auto", "tool_call", "rule"}:
            raise ValueError("planning_mode must be auto, tool_call or rule")
        return mode
