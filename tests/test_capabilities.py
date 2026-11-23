from __future__ import annotations

import unittest
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from cat_assistant.application.capabilities import (
    CapabilityExecutionContext,
    CapabilityRegistry,
)
from cat_assistant.application.orchestration import (
    RuleBasedTaskPlanner,
    ToolCallingTaskPlanner,
)
from cat_assistant.domain.capabilities import (
    Artifact,
    CapabilityDescriptor,
    CapabilityResult,
)
from cat_assistant.domain.models import (
    ChatMessage,
    ModelReply,
    ToolCall,
    ToolSpec,
    Utterance,
)
from cat_assistant.domain.plans import TaskStep
from cat_assistant.adapters.knowledge import demo_documents
from cat_assistant.adapters.telemetry import demo_snapshot
from cat_assistant.adapters.capabilities import _format_diagnosis, _scope_current_fault_hits


class IndicatorProvider:
    descriptor = CapabilityDescriptor(
        name="indicator.inspect",
        version="1.0.0",
        description="Inspect a warning indicator.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "required": ["color"]},
        kind="observe",
        source="test-plugin",
        trigger_terms=("指示灯",),
        auto_select=True,
        visible_in_response=True,
    )

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.start_count = 0
        self.stop_count = 0

    async def startup(self) -> None:
        self.started = True
        self.start_count += 1

    async def shutdown(self) -> None:
        self.stopped = True
        self.stop_count += 1

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "camera": "online"}

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del arguments, context, step
        return CapabilityResult(
            artifacts=(Artifact("indicator_observation", {"color": "red"}, self.descriptor.name),)
        )


class InvalidOutputProvider(IndicatorProvider):
    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del arguments, context, step
        return CapabilityResult(
            artifacts=(Artifact("indicator_observation", {"pattern": "flash"}, self.descriptor.name),)
        )


class AlarmProvider(IndicatorProvider):
    descriptor = CapabilityDescriptor(
        name="alarm.classify",
        version="1.0.0",
        description="Classify an audible machine alarm.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        kind="observe",
        source="test-plugin",
        produces=("alarm_observation",),
        visible_in_response=True,
    )


class HiddenPlumbingProvider(IndicatorProvider):
    """A registered-and-enabled capability that is withheld from the model menu."""

    descriptor = CapabilityDescriptor(
        name="telemetry.plumbing",
        version="1.0.0",
        description="内部管道能力，仅供规则规划器与依赖图使用。",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object"},
        kind="observe",
        source="test-plugin",
        model_selectable=False,
    )


class SelectingModel:
    def __init__(self) -> None:
        self.received_tools: tuple[ToolSpec, ...] = ()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        del messages
        self.received_tools = tuple(tools)
        names = {tool.capability: tool.name for tool in tools}
        return ModelReply(
            tool_calls=(
                ToolCall(names["indicator.inspect"], {"text": "指示灯和报警声"}),
                ToolCall(names["alarm.classify"], {"text": "指示灯和报警声"}),
            )
        )


class EmptySelectionModel:
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        del messages, tools
        return ModelReply(content="No tool selected")


class InvalidArgumentsModel:
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        del messages
        indicator = next(tool for tool in tools if tool.capability == "indicator.inspect")
        return ModelReply(tool_calls=(ToolCall(indicator.name, {}),))


class CapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_enabled_provider_starts_once_and_owner_shutdown_closes_it(self) -> None:
        registry = CapabilityRegistry()
        provider = IndicatorProvider()
        registry.register(provider, owner="vision-plugin")
        self.assertFalse(provider.started)

        await registry.start_all()
        await registry.start_all()
        self.assertTrue(provider.started)
        self.assertEqual(provider.start_count, 1)

        await registry.shutdown_owner("vision-plugin")
        self.assertTrue(provider.stopped)
        self.assertEqual(provider.stop_count, 1)

    async def test_started_owner_must_be_shutdown_before_unregister(self) -> None:
        registry = CapabilityRegistry()
        provider = IndicatorProvider()
        registry.register(provider, owner="vision-plugin")
        await registry.start_all()
        with self.assertRaisesRegex(RuntimeError, "shutdown_owner"):
            registry.unregister_owner("vision-plugin")
        await registry.shutdown_owner("vision-plugin")
        self.assertEqual(registry.unregister_owner("vision-plugin"), ("indicator.inspect",))

    async def test_tool_call_planner_exposes_enabled_capabilities_and_uses_model_selection(self) -> None:
        registry = CapabilityRegistry()
        registry.register(IndicatorProvider())
        registry.register(AlarmProvider())
        disabled = AlarmProvider()
        disabled.descriptor = CapabilityDescriptor(
            name="disabled.observe",
            version="1.0.0",
            description="Disabled test capability.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        registry.register(disabled, enabled=False)
        model = SelectingModel()
        planner = ToolCallingTaskPlanner(model, registry, RuleBasedTaskPlanner(registry))

        plan = await planner.build(
            Utterance("查看指示灯和报警声", "session", "cat-306-demo"),
            demo_snapshot(),
        )

        self.assertEqual(
            [step.capability for step in plan.steps],
            ["indicator.inspect", "alarm.classify"],
        )
        self.assertEqual(plan.steps[1].depends_on, ("model-1",))
        self.assertEqual(
            {tool.capability for tool in model.received_tools},
            {"indicator.inspect", "alarm.classify"},
        )
        self.assertTrue(all(len(tool.name) <= 64 for tool in model.received_tools))
        self.assertTrue(all("." not in tool.name for tool in model.received_tools))

    async def test_non_selectable_capability_is_hidden_from_model_but_stays_enabled(self) -> None:
        # A plumbing capability (model_selectable=False) must not appear on the
        # model's tool menu, yet must remain enabled and executable so the rule
        # planner and dependency graph can still use it.
        registry = CapabilityRegistry()
        registry.register(IndicatorProvider())
        registry.register(HiddenPlumbingProvider())

        tool_caps = {tool.capability for tool in registry.model_tools}
        enabled = {descriptor.name for descriptor in registry.enabled_descriptors}

        self.assertNotIn("telemetry.plumbing", tool_caps)
        self.assertIn("indicator.inspect", tool_caps)
        self.assertIn("telemetry.plumbing", enabled)
        self.assertIsNotNone(registry.provider("telemetry.plumbing"))

    async def test_model_tool_description_is_chinese_without_internal_jargon(self) -> None:
        # The description the model sees carries the capability's own text plus a
        # Chinese read-only/approval hint — never the old English scaffolding or
        # internal artifact vocabulary (Consumes/Produces), which is noise to a
        # small edge model.
        registry = CapabilityRegistry()
        registry.register(IndicatorProvider())
        description = registry.model_tools[0].description

        self.assertNotIn("Consumes", description)
        self.assertNotIn("Produces", description)
        self.assertNotIn("Capability", description)
        self.assertIn("只读", description)

    async def test_builtin_menu_is_trimmed_and_descriptions_are_chinese(self) -> None:
        # Guards the actual product config: which built-ins the model may pick and
        # that their model-facing descriptions are Chinese.
        from cat_assistant.adapters.capabilities import (
            ActionProposalProvider,
            DeterministicResponseProvider,
            DiagnosticAgentProvider,
            KnowledgeSearchProvider,
            MachineControlProvider,
            TelemetryReadProvider,
            TelemetrySummaryProvider,
            UserReportedObservationProvider,
        )

        self.assertIn("诊断", DiagnosticAgentProvider.descriptor.description)
        self.assertIn("摘要", TelemetrySummaryProvider.descriptor.description)
        # Plumbing / deterministic / niche capabilities are withheld from the model.
        self.assertFalse(TelemetryReadProvider.descriptor.model_selectable)
        self.assertFalse(DeterministicResponseProvider.descriptor.model_selectable)
        self.assertFalse(UserReportedObservationProvider.descriptor.model_selectable)
        # The five decision-relevant capabilities remain selectable.
        self.assertTrue(TelemetrySummaryProvider.descriptor.model_selectable)
        self.assertTrue(KnowledgeSearchProvider.descriptor.model_selectable)
        self.assertTrue(DiagnosticAgentProvider.descriptor.model_selectable)
        self.assertTrue(ActionProposalProvider.descriptor.model_selectable)
        self.assertTrue(MachineControlProvider.descriptor.model_selectable)

    async def test_tool_call_planner_falls_back_when_model_selects_nothing(self) -> None:
        registry = CapabilityRegistry()
        registry.register(IndicatorProvider())
        fallback = RuleBasedTaskPlanner(registry)
        planner = ToolCallingTaskPlanner(EmptySelectionModel(), registry, fallback)

        plan = await planner.build(
            Utterance("液压指示灯亮了", "session", "cat-306-demo"),
            demo_snapshot(),
        )

        self.assertEqual(plan.reason, "registry-driven request decomposition")
        self.assertIn("indicator.inspect", [step.capability for step in plan.steps])

    async def test_tool_call_planner_rejects_invalid_arguments_before_execution(self) -> None:
        registry = CapabilityRegistry()
        registry.register(IndicatorProvider())
        fallback = RuleBasedTaskPlanner(registry)
        planner = ToolCallingTaskPlanner(InvalidArgumentsModel(), registry, fallback)

        plan = await planner.build(
            Utterance("液压指示灯亮了", "session", "cat-306-demo"),
            demo_snapshot(),
        )

        self.assertEqual(plan.reason, "registry-driven request decomposition")

    async def test_planner_discovers_new_observation_provider_without_branch_change(self) -> None:
        registry = CapabilityRegistry()
        provider = IndicatorProvider()
        registry.register(provider, owner="vision-plugin")
        planner = RuleBasedTaskPlanner(registry)

        plan = await planner.build(
            Utterance("液压指示灯亮了，是什么原因？", "session", "cat-306-demo"),
            demo_snapshot(),
        )

        self.assertIn("indicator.inspect", [step.capability for step in plan.steps])
        discovered = next(step for step in plan.steps if step.capability == "indicator.inspect")
        self.assertEqual(discovered.arguments["text"], "液压指示灯亮了，是什么原因？")

    async def test_provider_lifecycle_health_and_execution_contract(self) -> None:
        registry = CapabilityRegistry()
        provider = IndicatorProvider()
        registry.register(provider, owner="vision-plugin", enabled=False)
        self.assertFalse(registry.is_enabled("indicator.inspect"))

        await registry.enable("indicator.inspect")
        self.assertTrue(provider.started)
        self.assertEqual((await registry.health("indicator.inspect"))["camera"], "online")
        step = TaskStep("vision", "indicator.inspect", {"text": "red light"})
        result = await registry.execute(
            step,
            CapabilityExecutionContext(
                Utterance("red light", "session", "cat-306-demo"),
                demo_snapshot(),
            ),
        )
        self.assertEqual(result.artifacts[0].artifact_type, "indicator_observation")

        await registry.disable("indicator.inspect")
        self.assertTrue(provider.stopped)

    async def test_registry_validates_input_and_output_contracts(self) -> None:
        registry = CapabilityRegistry()
        registry.register(IndicatorProvider())
        context = CapabilityExecutionContext(
            Utterance("light", "session", "cat-306-demo"),
            demo_snapshot(),
        )
        with self.assertRaisesRegex(ValueError, "missing capability arguments"):
            await registry.execute(TaskStep("vision", "indicator.inspect"), context)

        invalid = CapabilityRegistry()
        invalid.register(InvalidOutputProvider())
        with self.assertRaisesRegex(ValueError, "capability output is missing: color"):
            await invalid.execute(
                TaskStep("vision", "indicator.inspect", {"text": "red light"}),
                context,
            )


class CurrentFaultScopingTests(unittest.TestCase):
    def test_current_question_drops_manual_for_inactive_code(self) -> None:
        machine = replace(demo_snapshot(), fault_codes=("E123",))
        scoped = _scope_current_fault_hits(
            demo_documents(), machine, "当前存在哪些问题"
        )
        text = " ".join(hit.content for hit in scoped)
        self.assertIn("E123", text)
        self.assertNotIn("E456", text)

    def test_explicitly_requested_code_keeps_its_manual(self) -> None:
        machine = replace(demo_snapshot(), fault_codes=("E123",))
        scoped = _scope_current_fault_hits(
            demo_documents(), machine, "E456 怎么解决"
        )
        text = " ".join(hit.content for hit in scoped)
        self.assertIn("E456", text)
        self.assertIn("E123", text)


class DiagnosisFormattingTests(unittest.TestCase):
    def test_model_citations_replaced_with_single_canonical_source(self) -> None:
        # A small model glued a citation mid-sentence AND repeated 来源 lines;
        # normalisation removes every model-written citation and appends one
        # clean, canonical line built from the device-scoped hits.
        raw = (
            "结论：E45 冷却系统异常。\n"
            "建议：停机检查冷却液。来源：旧手册 §9\n"
            "来源：Cat 306 CR Service Manual §6.2"
        )
        out = _format_diagnosis(raw, "来源：Cat 950 GC §6.1；Cat 950 GC §7.3")
        self.assertEqual(out.count("来源"), 1)
        self.assertNotIn("旧手册", out)
        self.assertNotIn("Cat 306 CR", out)
        self.assertIn("停机检查冷却液。", out)
        self.assertTrue(out.endswith("来源：Cat 950 GC §6.1；Cat 950 GC §7.3"))

    def test_stray_citation_removed_when_no_evidence(self) -> None:
        # No retrieval this turn: an ungrounded model citation is dropped and
        # nothing is re-attached.
        out = _format_diagnosis("结论：设备正常。来源：不该出现的引用", None)
        self.assertEqual(out, "结论：设备正常。")


if __name__ == "__main__":
    unittest.main()
