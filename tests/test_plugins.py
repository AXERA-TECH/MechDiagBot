from __future__ import annotations

import unittest
from typing import Any

from cat_assistant.adapters.events import InMemoryEventStore
from cat_assistant.application.capabilities import CapabilityExecutionContext, CapabilityRegistry
from cat_assistant.application.events import EventRecorder
from cat_assistant.application.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    ServiceRegistry,
)
from cat_assistant.application.tools import ToolRegistry
from cat_assistant.domain.models import ToolExecutionContext, ToolSpec
from cat_assistant.domain.capabilities import CapabilityDescriptor, CapabilityResult
from cat_assistant.domain.plans import TaskStep


class PingTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="demo_ping",
            description="Return a plugin health response.",
            input_schema={"type": "object", "additionalProperties": False},
        )

    async def execute(self, arguments: dict[str, object], context: ToolExecutionContext) -> str:
        del arguments, context
        return "pong"


class DemoPlugin:
    manifest = PluginManifest("demo.plugin", "1.0.0")

    def register(self, context: PluginContext) -> None:
        context.tools.register(PingTool(), owner=self.manifest.name)

    def shutdown(self, context: PluginContext) -> None:
        del context


class PingCapability:
    descriptor = CapabilityDescriptor(
        name="demo.ping",
        version="1.0.0",
        description="Plugin capability health response.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object"},
    )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del arguments, context, step
        return CapabilityResult(content="pong")


class LifecycleCapability(PingCapability):
    descriptor = CapabilityDescriptor(
        name="demo.lifecycle",
        version="1.0.0",
        description="Plugin lifecycle test capability.",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object"},
    )

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    async def startup(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        self.stopped += 1


class CapabilityPlugin(DemoPlugin):
    manifest = PluginManifest("capability.plugin", "1.0.0")

    def register(self, context: PluginContext) -> None:
        assert context.capabilities is not None
        context.capabilities.register(PingCapability(), owner=self.manifest.name)


class LifecyclePlugin(DemoPlugin):
    manifest = PluginManifest("lifecycle.plugin", "1.0.0")

    def __init__(self) -> None:
        self.capability = LifecycleCapability()

    def register(self, context: PluginContext) -> None:
        assert context.capabilities is not None
        context.capabilities.register(self.capability, owner=self.manifest.name)


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def test_plugin_registers_and_unloads_owned_tools(self) -> None:
        store = InMemoryEventStore()
        manager = PluginManager(
            PluginContext(ToolRegistry(), ServiceRegistry(), EventRecorder(store))
        )

        manager.load_sync(DemoPlugin())
        self.assertEqual(manager.plugins[0].name, "demo.plugin")
        self.assertEqual(manager.context.tools.owner_of("demo_ping"), "demo.plugin")
        self.assertEqual([spec.name for spec in manager.context.tools.read_only_specs], ["demo_ping"])

        await manager.unload("demo.plugin")
        self.assertEqual(manager.context.tools.specs, ())

    async def test_plugin_dependency_must_be_loaded_first(self) -> None:
        class DependentPlugin(DemoPlugin):
            manifest = PluginManifest("dependent", "1.0.0", dependencies=("base",))

        manager = PluginManager(
            PluginContext(ToolRegistry(), ServiceRegistry(), EventRecorder(InMemoryEventStore()))
        )
        with self.assertRaises(RuntimeError):
            manager.load_sync(DependentPlugin())

    async def test_plugin_owned_capabilities_are_removed_on_unload(self) -> None:
        capabilities = CapabilityRegistry()
        manager = PluginManager(
            PluginContext(
                ToolRegistry(),
                ServiceRegistry(),
                EventRecorder(InMemoryEventStore()),
                capabilities=capabilities,
            )
        )
        manager.load_sync(CapabilityPlugin())
        self.assertEqual(capabilities.owner_of("demo.ping"), "capability.plugin")

        await manager.unload("capability.plugin")
        self.assertIsNone(capabilities.descriptor("demo.ping"))

    async def test_plugin_unload_stops_started_capabilities(self) -> None:
        capabilities = CapabilityRegistry()
        plugin = LifecyclePlugin()
        manager = PluginManager(
            PluginContext(
                ToolRegistry(),
                ServiceRegistry(),
                EventRecorder(InMemoryEventStore()),
                capabilities=capabilities,
            )
        )
        manager.load_sync(plugin)
        await capabilities.start_all()
        self.assertEqual(plugin.capability.started, 1)

        await manager.unload("lifecycle.plugin")
        self.assertEqual(plugin.capability.stopped, 1)


if __name__ == "__main__":
    unittest.main()
