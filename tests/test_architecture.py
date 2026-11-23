from __future__ import annotations

import asyncio
import unittest
from collections.abc import Sequence

from cat_assistant.adapters.events import InMemoryEventStore
from cat_assistant.adapters.capabilities import builtin_capability_providers
from cat_assistant.adapters.knowledge import InMemoryKnowledge, demo_documents
from cat_assistant.adapters.model import DemoRuleBasedModel
from cat_assistant.adapters.telemetry import InMemoryTelemetry, demo_snapshot
from cat_assistant.adapters.tools import MachineStatusTool, ManualSearchTool
from cat_assistant.application.events import EventRecorder
from cat_assistant.application.capabilities import CapabilityRegistry
from cat_assistant.application.context import ContextBuilder
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
from cat_assistant.application.voice import VoiceGateway
from cat_assistant.adapters.memory import InMemoryMemoryStore, InMemorySessionStore
from cat_assistant.domain.models import (
    ChatMessage,
    ModelReply,
    ResponseCategory,
    ToolCall,
    ToolSpec,
    TurnRecord,
    Utterance,
)


class ExplodingModel:
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        raise AssertionError("the deterministic lane must not invoke an LLM")


class EndlessToolModel:
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        return ModelReply(tool_calls=(ToolCall("get_machine_status", {}),))


class SlowModel:
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        del messages, tools
        await asyncio.sleep(0.05)
        return ModelReply(content="late")


class SlowTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slow_tool",
            description="A slow read-only test tool.",
            input_schema={"type": "object", "additionalProperties": False},
        )

    async def execute(self, arguments: dict[str, object], context: object) -> str:
        del arguments, context
        await asyncio.sleep(0.05)
        return "late"


def build_test_app(model: object = None, *, max_steps: int = 4):
    snapshot = demo_snapshot()
    telemetry = InMemoryTelemetry({snapshot.machine_id: snapshot})
    event_store = InMemoryEventStore()
    recorder = EventRecorder(event_store)
    knowledge = InMemoryKnowledge(demo_documents())
    tools = ToolRegistry((MachineStatusTool(), ManualSearchTool(knowledge)))
    runner = BoundedAgentRunner(
        model or DemoRuleBasedModel(),  # type: ignore[arg-type]
        tools,
        recorder,
        max_steps=max_steps,
    )
    sessions = InMemorySessionStore()
    memory = InMemoryMemoryStore()
    context_builder = ContextBuilder(sessions, memory)
    query_service = DeterministicQueryService()
    capabilities = CapabilityRegistry()
    for provider in builtin_capability_providers(
        knowledge=knowledge,
        query_service=query_service,
        agent_runner=runner,
        context_builder=context_builder,
    ):
        capabilities.register(provider)
    orchestrator = TaskOrchestrator(
        RuleBasedTaskPlanner(capabilities),
        PlanExecutor(capabilities, PolicyEngine(), recorder),
    )
    app = AgentLoop(
        telemetry,
        orchestrator,
        recorder,
        sessions,
        memory,
    )
    app.runner = runner
    app.tool_registry = tools
    app.capability_registry = capabilities
    return app, event_store


def request(text: str, session_id: str = "session-1") -> Utterance:
    return Utterance(text, session_id, "cat-306-demo", "operator-1")


class ArchitectureTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_timeout_returns_and_releases_turn(self) -> None:
        app, events = build_test_app(SlowModel())
        app.runner._model_call_timeout_seconds = 0.001

        response = await app.handle(request("分析一个液压异常"))

        self.assertEqual(response.category, ResponseCategory.ERROR)
        self.assertEqual(response.metadata["reason"], "model_timeout")
        self.assertTrue(any(event.event_type == "model/timeout" for event in events.events))

    async def test_tool_timeout_is_returned_as_tool_error(self) -> None:
        from cat_assistant.application.tools import ToolRegistry
        from cat_assistant.domain.models import ToolExecutionContext

        tools = ToolRegistry((SlowTool(),))
        result = await tools.execute(
            ToolCall("slow_tool", {}),
            ToolExecutionContext("session", "cat-306-demo", demo_snapshot()),
            timeout_seconds=0.001,
        )

        self.assertTrue(result.is_error)
        self.assertIn("timed out", result.content)

    async def test_status_query_bypasses_llm(self) -> None:
        app, events = build_test_app(ExplodingModel())

        response = await app.handle(request("下一次保养是什么时候？"))

        self.assertEqual(response.category, ResponseCategory.INFORMATION)
        self.assertIn("约剩 1.8 小时", response.text)
        plans = [event for event in events.events if event.event_type == "plan/created"]
        self.assertEqual(plans[0].payload["steps"], ["telemetry.read", "response.compose"])
        self.assertFalse(any(event.event_type == "intent/classified" for event in events.events))

    async def test_diagnostic_uses_planned_retrieval_and_cites_source(self) -> None:
        app, events = build_test_app()

        response = await app.handle(request("E123 故障码应该怎么检查？"))

        self.assertEqual(response.category, ResponseCategory.DIAGNOSTIC)
        self.assertIn("Service Manual", response.text)
        event_types = [event.event_type for event in events.events]
        plans = [event for event in events.events if event.event_type == "plan/created"]
        self.assertIn("knowledge.search", plans[0].payload["steps"])
        self.assertIn("diagnostic.agent", plans[0].payload["steps"])
        self.assertEqual(event_types[-1], "turn/completed")

    async def test_control_capability_stops_at_approval_policy(self) -> None:
        app, events = build_test_app(ExplodingModel())

        response = await app.handle(request("把 E-Ceiling 设置为 4.5 米"))

        self.assertEqual(response.category, ResponseCategory.APPROVAL_REQUIRED)
        self.assertTrue(response.requires_confirmation)
        self.assertFalse(any(event.event_type == "tool/called" for event in events.events))

    async def test_runner_stops_at_step_limit(self) -> None:
        app, events = build_test_app(EndlessToolModel(), max_steps=2)

        response = await app.handle(request("分析一个未知的液压异常"))

        self.assertEqual(response.category, ResponseCategory.ERROR)
        self.assertEqual(response.metadata["reason"], "max_steps")
        limits = [e for e in events.events if e.event_type == "agent/limit_reached"]
        self.assertEqual(len(limits), 1)
        self.assertEqual(limits[0].payload["max_steps"], 2)


class EchoGeneralModel:
    def __init__(self) -> None:
        self.messages: tuple[ChatMessage, ...] = ()
        self.tools: tuple[ToolSpec, ...] = ()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelReply:
        self.messages = tuple(messages)
        self.tools = tuple(tools)
        return ModelReply(content="这是通用回答。")


class ExplodingTelemetry:
    async def get_snapshot(self, machine_id: str) -> object:
        raise AssertionError(f"unbound turn must not read telemetry: {machine_id}")


class ExplodingOrchestrator:
    async def handle(self, utterance: Utterance, machine: object) -> object:
        raise AssertionError("unbound turn must not enter device orchestration")


class GeneralChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_unbound_turn_stays_in_general_lane(self) -> None:
        model = EchoGeneralModel()
        events = EventRecorder(InMemoryEventStore())
        sessions = InMemorySessionStore()
        memory = InMemoryMemoryStore()
        context_builder = ContextBuilder(sessions, memory)
        runner = BoundedAgentRunner(
            model,
            ToolRegistry(),
            events,
            context_builder=context_builder,
        )
        app = AgentLoop(
            ExplodingTelemetry(),  # type: ignore[arg-type]
            ExplodingOrchestrator(),  # type: ignore[arg-type]
            events,
            sessions,
            memory,
        )
        app.runner = runner
        await sessions.append_turn(
            TurnRecord(
                "general-session",
                "",
                "上一轮通用问题",
                "上一轮通用回答",
                "information",
                "operator-1",
            )
        )
        await sessions.append_turn(
            TurnRecord(
                "general-session",
                "cat-306-demo",
                "设备专属问题",
                "设备专属回答",
                "diagnostic",
                "operator-1",
            )
        )

        response = await app.handle(
            Utterance("你好", "general-session", "", "operator-1")
        )

        self.assertEqual(response.category, ResponseCategory.INFORMATION)
        self.assertEqual(response.metadata["mode"], "general")
        self.assertFalse(response.metadata["machine_attached"])
        self.assertEqual(model.tools, ())
        contents = [message.content for message in model.messages]
        self.assertTrue(any("未接入任何设备" in content for content in contents))
        self.assertIn("上一轮通用问题", contents)
        self.assertNotIn("设备专属问题", contents)
        turns = await sessions.recent_turns("general-session")
        self.assertEqual(turns[-1].machine_id, "")
        self.assertEqual(turns[-1].user_text, "你好")


class LowConfidenceRecognizer:
    async def transcribe(self, audio: bytes) -> tuple[str, float]:
        return "可能的噪声", 0.2


class TestSynthesizer:
    async def synthesize(self, text: str) -> bytes:
        return text.encode()


class VoiceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_low_confidence_audio_does_not_enter_agent_loop(self) -> None:
        app, events = build_test_app(ExplodingModel())
        gateway = VoiceGateway(LowConfidenceRecognizer(), TestSynthesizer(), app)

        reply = await gateway.handle_audio(
            b"noise",
            session_id="voice-session",
            machine_id="cat-306-demo",
        )

        self.assertEqual(reply.response.category, ResponseCategory.CLARIFICATION)
        self.assertEqual(events.events, [])


if __name__ == "__main__":
    unittest.main()
