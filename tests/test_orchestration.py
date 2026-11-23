from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from cat_assistant.adapters.capabilities import builtin_capability_providers
from cat_assistant.adapters.events import InMemoryEventStore
from cat_assistant.adapters.knowledge import InMemoryKnowledge, demo_documents
from cat_assistant.adapters.memory import InMemoryMemoryStore, InMemorySessionStore
from cat_assistant.adapters.telemetry import demo_snapshot
from cat_assistant.application.capabilities import (
    CapabilityRegistry,
    capability_tool_name,
)
from cat_assistant.application.context import ContextBuilder
from cat_assistant.application.events import EventRecorder
from cat_assistant.application.query import DeterministicQueryService
from cat_assistant.application.runner import BoundedAgentRunner
from cat_assistant.application.orchestration import (
    RuleBasedTaskPlanner,
    ToolCallingTaskPlanner,
)
from cat_assistant.application.tools import ToolRegistry
from cat_assistant.bootstrap import build_demo_app
from cat_assistant.domain.models import (
    ChatMessage,
    ModelReply,
    ResponseCategory,
    ToolCall,
    ToolSpec,
    Utterance,
)


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_composite_request_carries_state_and_knowledge_into_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_demo_app(Path(directory) / "events.jsonl")
            response = await app.handle(
                Utterance(
                    "查看当前状态，查询 E123 维修手册，给出处理方案",
                    "composite-session",
                    "cat-306-demo",
                    "operator-1",
                )
            )

            self.assertEqual(response.category, ResponseCategory.DIAGNOSTIC)
            self.assertIn("当前状态：怠速", response.text)
            self.assertIn("Service Manual", response.text)
            self.assertIn("建议执行方案", response.text)
            self.assertEqual(
                response.metadata["steps"],
                ["observe", "state", "retrieve", "diagnose", "propose"],
            )

    async def test_fault_status_question_triggers_manual_retrieval(self) -> None:
        # "问题" is not one of the planner's manual keywords, so this question
        # used to plan only observe→diagnose. knowledge.search now advertises the
        # fault vocabulary through auto_select trigger_terms, so discovery folds
        # a canonical "retrieve" node into the plan and the manual is searched.
        with tempfile.TemporaryDirectory() as directory:
            app = build_demo_app(Path(directory) / "events.jsonl")
            response = await app.handle(
                Utterance(
                    "机器存在什么问题吗",
                    "fault-status-session",
                    "cat-950gc-demo",
                    "operator-1",
                )
            )

            self.assertEqual(response.category, ResponseCategory.DIAGNOSTIC)
            self.assertIn("knowledge.search", response.metadata["capabilities"])
            self.assertIn("retrieve", response.metadata["steps"])
            # The evidence must reach the operator as a cited manual source, not
            # an ungrounded assertion.
            self.assertIn("Service Manual", response.text)

    async def test_control_request_runs_full_propose_approve_execute_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_demo_app(Path(directory) / "events.jsonl")
            response = await app.handle(
                Utterance(
                    "查看状态并停止发动机",
                    "control-session",
                    "cat-306-demo",
                    "operator-1",
                )
            )

            # The action is still a gated later step: it stops for approval.
            self.assertEqual(response.category, ResponseCategory.APPROVAL_REQUIRED)
            self.assertTrue(response.requires_confirmation)
            self.assertIn("需要操作员确认", response.text)
            self.assertIn("machine.control", response.metadata["capabilities"])

            # After approval the virtual Safety Gateway executes instead of
            # dead-ending, and the actuation becomes the next authoritative read.
            confirmed = await app.handle(
                Utterance("确认执行", "control-session", "cat-306-demo", "operator-1")
            )
            self.assertNotEqual(confirmed.category, ResponseCategory.ERROR)
            self.assertIn("虚拟安全网关", confirmed.text)
            self.assertIn("熄火", confirmed.text)
            # The confirmation reply is just the execution report: the state
            # summary and proposal were already shown on the approval turn.
            self.assertNotIn("建议执行方案", confirmed.text)
            self.assertNotIn("当前状态：怠速", confirmed.text)

            after = await app.get_machine_snapshot("cat-306-demo")
            self.assertFalse(after.engine_running)
            self.assertEqual(after.operating_state, "熄火")

    async def test_read_only_instruction_does_not_request_control_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_demo_app(Path(directory) / "events.jsonl")
            response = await app.handle(Utterance(
                "请只读检查 E45 和 F210，不要执行任何操作",
                "readonly-session", "cat-950gc-demo", "operator-1",
            ))
            self.assertFalse(response.requires_confirmation)
            self.assertNotIn("machine.control", response.metadata["capabilities"])

    async def test_conversational_message_omits_machine_state_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_demo_app(Path(directory) / "events.jsonl")
            response = await app.handle(
                Utterance(
                    "你还活着吗",
                    "smalltalk-session",
                    "cat-306-demo",
                    "operator-1",
                )
            )

            # An unrelated/conversational turn must not prepend the state summary.
            self.assertNotIn("telemetry.summarize", response.metadata["capabilities"])
            self.assertNotIn("state", response.metadata["steps"])
            self.assertNotIn("当前状态：", response.text)
            # It still reaches the bounded diagnostic agent for a real reply.
            self.assertIn("diagnostic.agent", response.metadata["capabilities"])

    async def test_capabilities_are_visible_to_runtime_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_demo_app(Path(directory) / "events.jsonl")
            self.assertIn("telemetry.read", {item.name for item in app.capability_specs})
            self.assertIn("machine.control", {item.name for item in app.capability_specs})

    async def test_reported_light_and_alarm_become_scoped_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_demo_app(Path(directory) / "events.jsonl")
            response = await app.handle(
                Utterance(
                    "设备指示灯亮，报警声响，是什么原因，怎么解决？",
                    "perception-session",
                    "cat-306-demo",
                    "operator-1",
                )
            )

            artifact_types = {item["artifact_type"] for item in response.metadata["artifacts"]}
            self.assertIn("indicator_report", artifact_types)
            self.assertIn("audio_alarm_report", artifact_types)
            self.assertIn("observation.user_report", response.metadata["capabilities"])


class SingleToolModel:
    """Planner double: always selects one capability tool with empty arguments."""

    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        del messages, tools
        return ModelReply(tool_calls=(ToolCall(self._tool_name, {}),))


class OneCallModel:
    """Planner double: emits exactly one tool call with the given arguments."""

    def __init__(
        self,
        tool_name: str,
        arguments: dict,
        *,
        finish_reason: str | None = None,
        usage: dict | None = None,
    ) -> None:
        self._tool_name = tool_name
        self._arguments = arguments
        self._finish_reason = finish_reason
        self._usage = usage

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        del messages, tools
        return ModelReply(
            tool_calls=(ToolCall(self._tool_name, self._arguments),),
            finish_reason=self._finish_reason,
            usage=self._usage,
        )


class MultiCallModel:
    """Planner double: emits a fixed list of tool calls in order."""

    def __init__(self, calls: Sequence[ToolCall]) -> None:
        self._calls = tuple(calls)

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        del messages, tools
        return ModelReply(tool_calls=self._calls, finish_reason="tool_calls")


class RaisingModel:
    """Planner double that fails if it is ever asked to generate.

    Used to assert the deterministic-fact fast path skips the model entirely.
    """

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        del messages, tools
        raise AssertionError("model.complete must not be called on the fast path")


class ToolCallingPlannerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _registry() -> CapabilityRegistry:
        recorder = EventRecorder(InMemoryEventStore())
        runner = BoundedAgentRunner(SingleToolModel(""), ToolRegistry(), recorder)
        context_builder = ContextBuilder(InMemorySessionStore(), InMemoryMemoryStore())
        capabilities = CapabilityRegistry()
        for provider in builtin_capability_providers(
            knowledge=InMemoryKnowledge(demo_documents()),
            query_service=DeterministicQueryService(),
            agent_runner=runner,
            context_builder=context_builder,
        ):
            capabilities.register(provider)
        return capabilities

    async def test_evidence_only_selection_is_repaired_with_diagnostic_terminal(self) -> None:
        capabilities = self._registry()
        store = InMemoryEventStore()
        # The model selects only telemetry.read: an evidence step with no reply.
        # Instead of discarding it and reverting to the rule planner, the planner
        # keeps the read and appends a terminal answer-producing capability, so the
        # plan still speaks to the operator.
        model = SingleToolModel(capability_tool_name("telemetry.read"))
        planner = ToolCallingTaskPlanner(
            model,
            capabilities,
            RuleBasedTaskPlanner(capabilities),
            events=EventRecorder(store),
        )

        plan = await planner.build(
            Utterance("机器有什么问题", "s", "cat-306-demo", "op"),
            demo_snapshot(),
        )

        caps = [step.capability for step in plan.steps]
        # The model's evidence pick is preserved and a visible diagnosis is
        # appended as the final step (diagnostic.agent beats a bare summary).
        self.assertEqual(caps, ["telemetry.read", "diagnostic.agent"])
        self.assertTrue(capabilities.descriptor(caps[-1]).visible_in_response)
        self.assertEqual(
            plan.reason, "llm tool-call capability selection (terminal answer appended)"
        )
        # The appended terminal runs last: it depends on every prior step, so it
        # receives their evidence artifacts, and it takes no arguments.
        terminal = plan.steps[-1]
        self.assertEqual(terminal.depends_on, (plan.steps[0].step_id,))
        self.assertEqual(terminal.arguments, {})
        # The repair is observable rather than silent.
        repaired = [e for e in store.events if e.event_type == "planning/llm_plan_repaired"]
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].payload["appended"], "diagnostic.agent")
        self.assertEqual(repaired[0].payload["selected_tools"], ["telemetry.read"])

    async def test_hallucinated_extra_argument_is_stripped_not_fatal(self) -> None:
        capabilities = self._registry()
        # telemetry.summarize takes no arguments; a real model still occasionally
        # adds a stray empty ``text``. Under additionalProperties:false that used
        # to fail validation and sink the whole LLM plan.
        tool = capability_tool_name("telemetry.summarize")
        planner = ToolCallingTaskPlanner(
            OneCallModel(tool, {"text": ""}),
            capabilities,
            RuleBasedTaskPlanner(capabilities),
        )

        # A diagnosis question, not a bare status one, so it bypasses the
        # deterministic-fact fast path and actually reaches the LLM planner.
        plan = await planner.build(
            Utterance("机器有什么问题", "s", "cat-306-demo", "op"), demo_snapshot()
        )

        # The plan is the model's (not a rule-based fallback), the stray key
        # was stripped rather than fatal, and a diagnostic turn still received
        # the conclusion the model omitted.
        self.assertEqual(
            plan.reason, "llm tool-call capability selection (diagnosis appended)"
        )
        self.assertEqual(
            [step.capability for step in plan.steps],
            ["telemetry.summarize", "diagnostic.agent"],
        )
        self.assertEqual(plan.steps[0].arguments, {})

    async def test_summary_and_retrieval_without_diagnosis_is_repaired(self) -> None:
        capabilities = self._registry()
        store = InMemoryEventStore()
        # Real-model failure observed in the debug console: for a follow-up
        # "怎么解决这个故障呢", the planner selected a visible status summary and
        # manual retrieval but omitted diagnostic.agent. The status summary
        # satisfied the old "has a visible answer" check, so the operator got a
        # repeated status line and never saw the retrieved evidence explained.
        planner = ToolCallingTaskPlanner(
            MultiCallModel(
                [
                    ToolCall(capability_tool_name("telemetry.summarize"), {}),
                    ToolCall(
                        capability_tool_name("knowledge.search"),
                        {"query": "E456 冷却液温度过高怎么处理"},
                    ),
                ]
            ),
            capabilities,
            RuleBasedTaskPlanner(capabilities),
            events=EventRecorder(store),
        )

        plan = await planner.build(
            Utterance(
                "怎么解决这个故障呢", "s", "cat-320-demo", "op"
            ),
            demo_snapshot(),
        )

        self.assertEqual(
            [step.capability for step in plan.steps],
            ["telemetry.summarize", "knowledge.search", "diagnostic.agent"],
        )
        self.assertEqual(
            plan.reason, "llm tool-call capability selection (diagnosis appended)"
        )
        # The appended diagnosis consumes both the status summary and the
        # retrieval artifacts.
        self.assertEqual(plan.steps[-1].arguments, {})
        self.assertEqual(
            plan.steps[-1].depends_on, ("model-1", "model-2")
        )
        repaired = [e for e in store.events if e.event_type == "planning/llm_plan_repaired"]
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].payload["reason"], "appended_diagnosis")
        self.assertEqual(
            repaired[0].payload["selected_tools"],
            ["telemetry.summarize", "knowledge.search"],
        )

    async def test_simple_fact_question_skips_llm_planning(self) -> None:
        capabilities = self._registry()
        store = InMemoryEventStore()
        # A bare status/fact question ("下次保养时间") is answered deterministically
        # by the rule planner's response.compose branch, so the tool-calling
        # planner must NOT spend a (slow) model call on it. RaisingModel fails the
        # test if complete() is ever invoked.
        planner = ToolCallingTaskPlanner(
            RaisingModel(),
            capabilities,
            RuleBasedTaskPlanner(capabilities),
            events=EventRecorder(store),
        )

        plan = await planner.build(
            Utterance("下次保养时间", "s", "cat-306-demo", "op"), demo_snapshot()
        )

        # The rule plan is used verbatim: telemetry.read -> response.compose.
        self.assertIn("response.compose", [step.capability for step in plan.steps])
        self.assertEqual(plan.reason, "registry-driven request decomposition")
        # The skip is observable rather than silent.
        fast = [e for e in store.events if e.event_type == "planning/fast_path"]
        self.assertEqual(len(fast), 1)
        self.assertEqual(fast[0].payload["reason"], "deterministic_fact")

    async def test_rejected_plan_records_observable_event(self) -> None:
        capabilities = self._registry()
        store = InMemoryEventStore()
        # knowledge.search requires ``query``; the model omits it and there is no
        # stray key to strip, so validation still fails and the plan falls back —
        # but that fallback is now observable instead of silent.
        tool = capability_tool_name("knowledge.search")
        planner = ToolCallingTaskPlanner(
            OneCallModel(tool, {}),
            capabilities,
            RuleBasedTaskPlanner(capabilities),
            events=EventRecorder(store),
        )

        plan = await planner.build(
            Utterance("查一下手册", "s", "cat-306-demo", "op"), demo_snapshot()
        )

        self.assertEqual(plan.reason, "registry-driven request decomposition")
        rejected = [e for e in store.events if e.event_type == "planning/llm_plan_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].payload["reason"], "invalid_arguments")
        self.assertEqual(rejected[0].payload["capability"], "knowledge.search")

    async def test_unrepairable_evidence_only_plan_falls_back_and_logs_metadata(self) -> None:
        capabilities = self._registry()
        # Remove every capability that could serve as an appended terminal answer
        # (diagnostic.agent / telemetry.summarize). With none left, an
        # evidence-only selection can no longer be repaired and must fall back to
        # the deterministic planner — and that rejection still carries the model's
        # finish_reason and usage, so the log separates deliberate under-selection
        # from truncation without a trace dive.
        await capabilities.disable("diagnostic.agent")
        await capabilities.disable("telemetry.summarize")
        store = InMemoryEventStore()
        planner = ToolCallingTaskPlanner(
            OneCallModel(
                capability_tool_name("telemetry.read"),
                {},
                finish_reason="tool_calls",
                usage={"total_tokens": 120},
            ),
            capabilities,
            RuleBasedTaskPlanner(capabilities),
            events=EventRecorder(store),
        )

        plan = await planner.build(
            Utterance("机器有什么问题", "s", "cat-306-demo", "op"), demo_snapshot()
        )

        # It fell back to the deterministic planner rather than the LLM plan.
        self.assertEqual(plan.reason, "registry-driven request decomposition")
        rejected = [e for e in store.events if e.event_type == "planning/llm_plan_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].payload["reason"], "no_visible_answer")
        self.assertEqual(rejected[0].payload["finish_reason"], "tool_calls")
        self.assertEqual(rejected[0].payload["usage"], {"total_tokens": 120})

    async def test_synonym_argument_key_is_rerouted_not_dropped(self) -> None:
        capabilities = self._registry()
        store = InMemoryEventStore()
        # Model uses the wrong-but-common key ``text`` for knowledge.search (whose
        # slot is ``query``), paired with a visible answer step. The value must be
        # rerouted to ``query`` and the plan survive as the model's own — not get
        # stripped and rejected for a "missing query" it actually supplied.
        planner = ToolCallingTaskPlanner(
            MultiCallModel(
                [
                    ToolCall(capability_tool_name("knowledge.search"), {"text": "E123 检修步骤"}),
                    ToolCall(capability_tool_name("diagnostic.agent"), {}),
                ]
            ),
            capabilities,
            RuleBasedTaskPlanner(capabilities),
            events=EventRecorder(store),
        )

        plan = await planner.build(
            Utterance("怎么修", "s", "cat-306-demo", "op"), demo_snapshot()
        )

        self.assertEqual(plan.reason, "llm tool-call capability selection")
        search = next(s for s in plan.steps if s.capability == "knowledge.search")
        self.assertEqual(search.arguments, {"query": "E123 检修步骤"})
        coerced = [e for e in store.events if e.event_type == "planning/llm_args_coerced"]
        self.assertEqual(len(coerced), 1)
        self.assertEqual(coerced[0].payload["capability"], "knowledge.search")
        self.assertEqual(coerced[0].payload["remapped"], [{"from": "text", "to": "query"}])

    async def test_rejection_event_lists_full_model_selection(self) -> None:
        capabilities = self._registry()
        store = InMemoryEventStore()
        # A sensible chain that botches an argument (knowledge.search with no query
        # and no synonym to reroute) must still log the WHOLE selection, not merely
        # the tool that tripped validation — so the full intent stays visible.
        planner = ToolCallingTaskPlanner(
            MultiCallModel(
                [
                    ToolCall(capability_tool_name("telemetry.read"), {}),
                    ToolCall(capability_tool_name("knowledge.search"), {}),
                ]
            ),
            capabilities,
            RuleBasedTaskPlanner(capabilities),
            events=EventRecorder(store),
        )

        await planner.build(
            Utterance("查手册", "s", "cat-306-demo", "op"), demo_snapshot()
        )

        rejected = [e for e in store.events if e.event_type == "planning/llm_plan_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].payload["reason"], "invalid_arguments")
        self.assertEqual(
            rejected[0].payload["selected_tools"], ["telemetry.read", "knowledge.search"]
        )


if __name__ == "__main__":
    unittest.main()
