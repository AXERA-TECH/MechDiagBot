from __future__ import annotations

import dataclasses
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from cat_assistant.adapters.capabilities import (
    MachineControlProvider,
    VerificationProvider,
    builtin_capability_providers,
)
from cat_assistant.adapters.control import SimulatedMachineControl
from cat_assistant.adapters.events import InMemoryEventStore
from cat_assistant.adapters.knowledge import InMemoryKnowledge, demo_documents
from cat_assistant.adapters.memory import InMemoryMemoryStore, InMemorySessionStore
from cat_assistant.adapters.model import DemoRuleBasedModel
from cat_assistant.adapters.telemetry import InMemoryTelemetry, demo_snapshots
from cat_assistant.adapters.tools import MachineStatusTool, ManualSearchTool
from cat_assistant.application.approvals import (
    ConfirmationKind,
    is_standalone_cancel,
    is_standalone_confirmation,
    parse_confirmation,
)
from cat_assistant.application.capabilities import CapabilityRegistry
from cat_assistant.application.context import ContextBuilder
from cat_assistant.application.events import EventRecorder
from cat_assistant.application.intent import ControlIntent, control_command
from cat_assistant.application.loop import AgentLoop
from cat_assistant.application.orchestration import (
    PlanExecutor,
    PolicyEngine,
    RuleBasedTaskPlanner,
    TaskOrchestrator,
)
from cat_assistant.application.query import DeterministicQueryService
from cat_assistant.application.runner import BoundedAgentRunner
from cat_assistant.application.tools import ToolRegistry
from cat_assistant.domain.models import (
    ControlOutcome,
    MachineSnapshot,
    ResponseCategory,
    Utterance,
)
from cat_assistant.domain.plans import TaskPlan, TaskStep


FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_app(*, ttl: float = 300.0, clock=None):
    telemetry = InMemoryTelemetry(dict(demo_snapshots()))
    event_store = InMemoryEventStore()
    events = EventRecorder(event_store)
    sessions = InMemorySessionStore()
    memory = InMemoryMemoryStore()
    knowledge = InMemoryKnowledge(demo_documents())
    tools = ToolRegistry((MachineStatusTool(), ManualSearchTool(knowledge)))
    context_builder = ContextBuilder(sessions, memory, history_limit=4, max_chars=6000)
    runner = BoundedAgentRunner(
        DemoRuleBasedModel(),
        tools,
        events,
        max_steps=2,
        context_builder=context_builder,
    )
    capabilities = CapabilityRegistry()
    for provider in builtin_capability_providers(
        knowledge=knowledge,
        query_service=DeterministicQueryService(),
        agent_runner=runner,
        context_builder=context_builder,
        machine_control=SimulatedMachineControl(telemetry),
    ):
        capabilities.register(provider, owner="cat.core")
    executor = PlanExecutor(
        capabilities, PolicyEngine(), events, telemetry=telemetry
    )
    orchestrator = TaskOrchestrator(
        RuleBasedTaskPlanner(capabilities),
        executor,
        events=events,
        approval_ttl_seconds=ttl,
        clock=clock,
    )
    app = AgentLoop(telemetry, orchestrator, events, sessions, memory)
    app.runner = runner
    app.capability_registry = capabilities
    app.tool_registry = tools
    return SimpleNamespace(
        app=app,
        telemetry=telemetry,
        event_store=event_store,
        capabilities=capabilities,
    )


def request(
    text: str,
    *,
    session_id: str = "approval-session",
    machine_id: str = "cat-306-demo",
    operator_id: str = "operator-1",
) -> Utterance:
    return Utterance(text, session_id, machine_id, operator_id)


class ConfirmationParsingTests(unittest.TestCase):
    def test_positive_replies_are_confirmations(self) -> None:
        for text in ("确认执行", "确认", "同意", "同意执行", "yes", "Yes!", "proceed"):
            self.assertIs(parse_confirmation(text), ConfirmationKind.CONFIRM, text)

    def test_casual_acknowledgements_are_not_approvals(self) -> None:
        for text in ("好", "好的", "ok", "可以"):
            self.assertIs(parse_confirmation(text), ConfirmationKind.NONE, text)

    def test_negated_replies_cancel_even_when_they_contain_confirm_words(self) -> None:
        for text in (
            "我不确认，请不要执行",
            "确认取消",
            "不要执行",
            "取消执行",
            "no",
            "not approved",
            "do not proceed",
        ):
            self.assertIs(parse_confirmation(text), ConfirmationKind.CANCEL, text)

    def test_substrings_are_not_confirmations(self) -> None:
        for text in ("yesterday", "确认一下油量", "停止发动机", "现在状态如何"):
            self.assertIs(parse_confirmation(text), ConfirmationKind.NONE, text)

    def test_standalone_checks_distinguish_replies_from_requests(self) -> None:
        self.assertTrue(is_standalone_cancel("取消执行"))
        self.assertFalse(is_standalone_cancel("检查故障，不要执行任何操作"))
        self.assertTrue(is_standalone_confirmation("确认执行"))
        self.assertFalse(is_standalone_confirmation("好"))


class ControlIntentTests(unittest.TestCase):
    def test_question_and_negation_do_not_become_control_commands(self) -> None:
        self.assertIs(control_command("停机会怎么样？"), ControlIntent.QUESTION)
        self.assertIs(control_command("启动了么"), ControlIntent.QUESTION)
        self.assertIs(control_command("那么停止发动机吧"), ControlIntent.STOP)
        self.assertIs(control_command("停机"), ControlIntent.STOP)
        self.assertIs(control_command("启动发动机"), ControlIntent.START)
        self.assertIs(control_command("停止发动机"), ControlIntent.STOP)
        self.assertIs(control_command("不要停机"), ControlIntent.NONE)
        self.assertIs(control_command("无法停机"), ControlIntent.NONE)
        self.assertIs(control_command("启动并停止发动机"), ControlIntent.CONFLICT)


class ApprovalBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmation_must_come_from_the_operator_who_requested_it(self) -> None:
        fixture = make_app()
        proposed = await fixture.app.handle(request("停止发动机"))
        self.assertIs(proposed.category, ResponseCategory.APPROVAL_REQUIRED)
        self.assertIn("approval", proposed.metadata)
        self.assertEqual(
            proposed.metadata["approval"]["operator_id"], "operator-1"
        )

        other_operator = await fixture.app.handle(
            request("确认执行", operator_id="operator-2")
        )
        self.assertIs(other_operator.category, ResponseCategory.CLARIFICATION)
        self.assertEqual(
            other_operator.metadata["reason"], "approval_operator_mismatch"
        )
        self.assertTrue(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )

        # The original requester still holds the (single) pending approval.
        confirmed = await fixture.app.handle(request("确认执行"))
        self.assertIs(confirmed.category, ResponseCategory.ACTION_COMPLETED)
        self.assertFalse(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )
        event_types = [event.event_type for event in fixture.event_store.events]
        self.assertIn("approval/operator_mismatch", event_types)
        self.assertEqual(event_types.count("approval/confirmed"), 1)

    async def test_confirmation_is_bound_to_the_requested_machine(self) -> None:
        fixture = make_app()
        await fixture.app.handle(request("停止发动机", machine_id="cat-306-demo"))

        mismatch = await fixture.app.handle(
            request("确认执行", machine_id="cat-950gc-demo")
        )
        self.assertIs(mismatch.category, ResponseCategory.CLARIFICATION)
        self.assertEqual(mismatch.metadata["reason"], "approval_machine_mismatch")
        self.assertTrue(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )

    async def test_negated_confirmation_cancels_the_pending_action(self) -> None:
        fixture = make_app()
        await fixture.app.handle(request("停止发动机"))

        cancelled = await fixture.app.handle(request("我不确认，请不要执行"))
        self.assertIs(cancelled.category, ResponseCategory.INFORMATION)
        self.assertEqual(cancelled.metadata["reason"], "approval_cancelled")
        self.assertTrue(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )

        # Once cancelled, a bare confirmation has nothing to run.
        after = await fixture.app.handle(request("确认执行"))
        self.assertIs(after.category, ResponseCategory.INFORMATION)
        self.assertEqual(after.metadata["reason"], "no_pending_approval")
        self.assertTrue(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )

    async def test_embedded_dont_execute_also_cancels_a_pending_action(self) -> None:
        fixture = make_app()
        await fixture.app.handle(request("停止发动机"))

        cancelled = await fixture.app.handle(
            request("先检查一下，不要执行任何操作")
        )
        self.assertIs(cancelled.category, ResponseCategory.INFORMATION)
        self.assertEqual(cancelled.metadata["reason"], "approval_cancelled")
        self.assertTrue(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )

    async def test_approval_expires(self) -> None:
        now = [FIXED_NOW]
        fixture = make_app(ttl=60, clock=lambda: now[0])
        await fixture.app.handle(request("停止发动机"))

        now[0] += timedelta(seconds=61)
        expired = await fixture.app.handle(request("确认执行"))

        self.assertIs(expired.category, ResponseCategory.CLARIFICATION)
        self.assertEqual(expired.metadata["reason"], "approval_expired")
        self.assertTrue(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )
        self.assertIn(
            "approval/expired",
            [event.event_type for event in fixture.event_store.events],
        )

    async def test_machine_state_change_invalidates_the_approval(self) -> None:
        fixture = make_app()
        await fixture.app.handle(request("停止发动机"))
        snapshot = await fixture.telemetry.get_snapshot("cat-306-demo")
        # The machine changes state while the approval is waiting: an external
        # stop, a fault clearing, or another operator acting.
        fixture.telemetry.apply_snapshot(
            dataclasses.replace(snapshot, engine_running=False, operating_state="熄火")
        )

        stale = await fixture.app.handle(request("确认执行"))

        self.assertIs(stale.category, ResponseCategory.CLARIFICATION)
        self.assertEqual(stale.metadata["reason"], "approval_state_changed")
        event_types = [event.event_type for event in fixture.event_store.events]
        self.assertIn("approval/state_changed", event_types)
        self.assertNotIn("approval/confirmed", event_types)

    async def test_a_new_request_supersedes_the_pending_plan(self) -> None:
        fixture = make_app()
        await fixture.app.handle(request("停止发动机"))

        unrelated = await fixture.app.handle(request("当前油量是多少"))
        self.assertIs(unrelated.category, ResponseCategory.INFORMATION)
        self.assertNotIn(
            "machine.control", unrelated.metadata.get("capabilities", [])
        )

        confirmed = await fixture.app.handle(request("确认执行"))
        self.assertIs(confirmed.category, ResponseCategory.INFORMATION)
        self.assertEqual(confirmed.metadata["reason"], "no_pending_approval")
        self.assertTrue(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )
        self.assertIn(
            "approval/superseded",
            [event.event_type for event in fixture.event_store.events],
        )

    async def test_casual_acknowledgement_does_not_execute_a_pending_action(self) -> None:
        fixture = make_app()
        await fixture.app.handle(request("停止发动机"))

        response = await fixture.app.handle(request("好的"))
        self.assertIsNot(response.category, ResponseCategory.ACTION_COMPLETED)
        self.assertTrue(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )
        # The ambiguous reply superseded the plan instead of keeping it alive.
        self.assertIn(
            "approval/superseded",
            [event.event_type for event in fixture.event_store.events],
        )

    async def test_clearing_the_session_drops_a_pending_approval(self) -> None:
        fixture = make_app()
        await fixture.app.handle(request("停止发动机"))

        await fixture.app.clear_session("approval-session")
        confirmed = await fixture.app.handle(request("确认执行"))

        self.assertIs(confirmed.category, ResponseCategory.INFORMATION)
        self.assertEqual(confirmed.metadata["reason"], "no_pending_approval")
        self.assertTrue(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )
        rejected = [
            event
            for event in fixture.event_store.events
            if event.event_type == "approval/rejected"
        ]
        self.assertEqual(rejected[0].payload["reason"], "session_cleared")

    async def test_negated_and_question_control_requests_never_ask_for_approval(self) -> None:
        fixture = make_app()
        for text in ("不要停机", "请不要启动发动机", "停机会怎么样？", "停止发动机吗？"):
            response = await fixture.app.handle(request(text, session_id=f"s-{text}"))
            self.assertIsNot(response.category, ResponseCategory.APPROVAL_REQUIRED, text)
            self.assertNotIn(
                "machine.control", response.metadata.get("capabilities", []), text
            )
            self.assertTrue(
                (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running,
                text,
            )


class ClosedLoopVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_approved_action_is_followed_by_a_live_verification_step(self) -> None:
        fixture = make_app()
        proposed = await fixture.app.handle(request("停止发动机"))
        self.assertIs(proposed.category, ResponseCategory.APPROVAL_REQUIRED)

        confirmed = await fixture.app.handle(request("确认执行"))

        self.assertIs(confirmed.category, ResponseCategory.ACTION_COMPLETED)
        self.assertEqual(confirmed.metadata["capabilities"][-2:], [
            "machine.control",
            "verify.state",
        ])
        self.assertIn("复核结果", confirmed.text)
        self.assertIn("动作已生效", confirmed.text)
        self.assertIn("一致", confirmed.text)
        self.assertNotIn("建议执行方案", confirmed.text)
        types = {item["artifact_type"] for item in confirmed.metadata["artifacts"]}
        self.assertIn("action_result", types)
        self.assertIn("verification_report", types)
        self.assertIs(
            confirmed.metadata["step_metadata"]["verify"]["matched"], True
        )
        self.assertFalse(
            (await fixture.telemetry.get_snapshot("cat-306-demo")).engine_running
        )

    async def test_verification_detects_a_gateway_that_did_not_change_the_machine(self) -> None:
        class GhostControl:
            async def apply(self, request: str, *, machine: MachineSnapshot):
                del request
                # Reports a state change to the plan, but the live telemetry
                # never moves -- exactly what a stuck actuator looks like.
                return ControlOutcome(
                    accepted=True,
                    summary="安全网关已接受指令。",
                    machine=dataclasses.replace(
                        machine, engine_running=False, operating_state="熄火"
                    ),
                )

        machine = demo_snapshots()["cat-306-demo"]
        telemetry = InMemoryTelemetry({machine.machine_id: machine})
        registry = CapabilityRegistry()
        registry.register(MachineControlProvider(GhostControl()))
        registry.register(VerificationProvider())
        executor = PlanExecutor(
            registry,
            PolicyEngine(),
            EventRecorder(InMemoryEventStore()),
            telemetry=telemetry,
        )
        plan = TaskPlan(
            steps=(
                TaskStep("act", "machine.control", {"request": "停止发动机"}),
                TaskStep("verify", "verify.state", {}, ("act",)),
            )
        )

        response = await executor.execute(
            plan,
            utterance=request("停止发动机"),
            machine=machine,
            approved=True,
        )

        self.assertIs(response.category, ResponseCategory.ACTION_COMPLETED)
        self.assertIn("不一致", response.text)
        self.assertIs(
            response.metadata["step_metadata"]["verify"]["matched"], False
        )
        self.assertTrue((await telemetry.get_snapshot(machine.machine_id)).engine_running)

    async def test_verification_reports_when_live_telemetry_is_unavailable(self) -> None:
        machine = demo_snapshots()["cat-306-demo"]
        telemetry = InMemoryTelemetry({machine.machine_id: machine})
        registry = CapabilityRegistry()
        registry.register(MachineControlProvider(SimulatedMachineControl(telemetry)))
        registry.register(VerificationProvider())
        # No telemetry port on the executor: verification must say it could not
        # read live state instead of presenting the action's own report as proof.
        executor = PlanExecutor(
            registry, PolicyEngine(), EventRecorder(InMemoryEventStore())
        )
        plan = TaskPlan(
            steps=(
                TaskStep("act", "machine.control", {"request": "停止发动机"}),
                TaskStep("verify", "verify.state", {}, ("act",)),
            )
        )

        response = await executor.execute(
            plan,
            utterance=request("停止发动机"),
            machine=machine,
            approved=True,
        )

        self.assertIn("无法读取动作后的实时遥测", response.text)
        self.assertIsNone(response.metadata["step_metadata"]["verify"]["matched"])

    async def test_control_adapter_refuses_negated_question_and_conflicting_text(self) -> None:
        snapshot = demo_snapshots()["cat-306-demo"]
        telemetry = InMemoryTelemetry({snapshot.machine_id: snapshot})
        executor = SimulatedMachineControl(telemetry)

        negated = await executor.apply("不要停止发动机", machine=snapshot)
        question = await executor.apply("停机会怎么样？", machine=snapshot)
        conflict = await executor.apply("启动并停止发动机", machine=snapshot)

        for outcome in (negated, question, conflict):
            self.assertFalse(outcome.accepted)
            self.assertIsNone(outcome.machine)
        self.assertEqual(await telemetry.get_snapshot(snapshot.machine_id), snapshot)


if __name__ == "__main__":
    unittest.main()
