from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from cat_assistant.adapters.events import InMemoryEventStore
from cat_assistant.adapters.capabilities import builtin_capability_providers
from cat_assistant.adapters.knowledge import InMemoryKnowledge, demo_documents
from cat_assistant.adapters.memory import (
    InMemoryMemoryStore,
    InMemorySessionStore,
    JsonlSessionStore,
)
from cat_assistant.adapters.telemetry import InMemoryTelemetry, demo_snapshot
from cat_assistant.application.context import ContextBuilder
from cat_assistant.application.events import EventRecorder
from cat_assistant.application.capabilities import CapabilityRegistry
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
    ChatMessage,
    MemoryRecord,
    ModelReply,
    ResponseCategory,
    TurnRecord,
    Utterance,
)


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[ChatMessage, ...]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[object],
    ) -> ModelReply:
        del tools
        self.calls.append(tuple(messages))
        return ModelReply(content="已记录本轮诊断回答")


def make_app(model: RecordingModel, *, memory: InMemoryMemoryStore | None = None) -> AgentLoop:
    snapshot = demo_snapshot()
    telemetry = InMemoryTelemetry({snapshot.machine_id: snapshot})
    recorder = EventRecorder(InMemoryEventStore())
    sessions = InMemorySessionStore()
    memory_store = memory or InMemoryMemoryStore()
    context_builder = ContextBuilder(sessions, memory_store)
    runner = BoundedAgentRunner(model, ToolRegistry(), recorder)
    query_service = DeterministicQueryService()
    capabilities = CapabilityRegistry()
    for provider in builtin_capability_providers(
        knowledge=InMemoryKnowledge(demo_documents()),
        query_service=query_service,
        agent_runner=runner,
        context_builder=context_builder,
    ):
        capabilities.register(provider)
    app = AgentLoop(
        telemetry,
        TaskOrchestrator(
            RuleBasedTaskPlanner(capabilities),
            PlanExecutor(capabilities, PolicyEngine(), recorder),
        ),
        recorder,
        session_store=sessions,
        memory=memory_store,
    )
    app.runner = runner
    app.capability_registry = capabilities
    return app


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostic_injects_retrieved_evidence_without_orphan_tool_message(self) -> None:
        model = RecordingModel()
        app = make_app(model)

        response = await app.handle(
            Utterance("E123 故障码应该怎么检查？", "evidence-session", "cat-306-demo", "op")
        )

        self.assertEqual(response.category, ResponseCategory.DIAGNOSTIC)
        self.assertTrue(model.calls)
        messages = model.calls[-1]
        self.assertFalse(any(message.role == "tool" for message in messages))
        self.assertTrue(
            any(
                message.role == "system"
                and message.content.startswith("以下为检索到的手册证据")
                for message in messages
            )
        )

    async def test_diagnostic_context_reuses_recent_turns(self) -> None:
        model = RecordingModel()
        app = make_app(model)

        await app.handle(Utterance("第一个诊断问题", "session", "cat-306-demo", "op"))
        response = await app.handle(Utterance("继续分析", "session", "cat-306-demo", "op"))

        self.assertEqual(response.category, ResponseCategory.DIAGNOSTIC)
        second = model.calls[1]
        self.assertIn("第一个诊断问题", [message.content for message in second])
        self.assertTrue(
            any("已记录本轮诊断回答" in message.content for message in second)
        )

    async def test_memory_is_scoped_to_machine_and_operator(self) -> None:
        memory = InMemoryMemoryStore(
            (
                MemoryRecord("operator prefers concise answers", operator_id="op"),
                MemoryRecord("other machine fact", machine_id="other"),
            )
        )
        builder = ContextBuilder(InMemorySessionStore(), memory)
        messages = await builder.build(
            session_id="session",
            user_text="operator prefers concise answers",
            machine=demo_snapshot(),
            operator_id="op",
        )

        joined = "\n".join(message.content for message in messages)
        self.assertIn("operator prefers concise answers", joined)
        self.assertNotIn("other machine fact", joined)

    async def test_context_budget_keeps_current_request(self) -> None:
        sessions = InMemorySessionStore()
        await sessions.append_turn(
            # Long history is deliberately not allowed to crowd out the current request.
            TurnRecord(
                session_id="session",
                machine_id="cat-306-demo",
                user_text="old " * 500,
                assistant_text="answer " * 500,
                category="diagnostic",
            )
        )
        builder = ContextBuilder(sessions, InMemoryMemoryStore(), max_chars=1_000)
        messages = await builder.build(
            session_id="session",
            user_text="CURRENT_REQUEST",
            machine=demo_snapshot(),
        )

        self.assertEqual(messages[-1].role, "user")
        self.assertIn("CURRENT_REQUEST", messages[-1].content)


class SessionStoreDeletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_memory_delete_removes_turns_and_summary(self) -> None:
        store = InMemorySessionStore()
        await store.append_turn(
            TurnRecord(
                session_id="a",
                machine_id="m",
                user_text="u",
                assistant_text="a",
                category="information",
            )
        )
        await store.set_summary("a", "summary")

        removed = await store.delete_session("a")

        self.assertEqual(removed, 1)
        self.assertEqual(await store.recent_turns("a"), ())
        self.assertIsNone(await store.get_summary("a"))

    async def test_jsonl_delete_rewrites_file_and_keeps_other_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.jsonl"
            store = JsonlSessionStore(path)
            await store.append_turn(
                TurnRecord(
                    session_id="keep",
                    machine_id="m",
                    user_text="k",
                    assistant_text="k",
                    category="information",
                )
            )
            await store.append_turn(
                TurnRecord(
                    session_id="drop",
                    machine_id="m",
                    user_text="d",
                    assistant_text="d",
                    category="information",
                )
            )
            await store.set_summary("drop", "to be removed")
            await store.set_summary("keep", "stays")

            removed = await store.delete_session("drop")

            self.assertEqual(removed, 1)
            self.assertEqual(await store.recent_turns("drop"), ())
            self.assertEqual(
                [turn.user_text for turn in await store.recent_turns("keep")],
                ["k"],
            )
            self.assertIsNone(await store.get_summary("drop"))
            self.assertEqual(await store.get_summary("keep"), "stays")

            # The deletion is persisted: a store reopened on the same files agrees.
            reopened = JsonlSessionStore(path)
            self.assertEqual(await reopened.recent_turns("drop"), ())
            self.assertEqual(
                [turn.user_text for turn in await reopened.recent_turns("keep")],
                ["k"],
            )


if __name__ == "__main__":
    unittest.main()
