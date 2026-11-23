from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cat_assistant.adapters.tracing import RecordingTracer, TracedLanguageModel
from cat_assistant.application.tracing import bind, current_node, span, unbind
from cat_assistant.bootstrap import _tracer_from_env, build_demo_app
from cat_assistant.domain.models import ChatMessage, ModelReply, ToolSpec, Utterance


class _FakeModel:
    model = "fake-model-1"

    async def complete(self, messages, tools):
        del messages, tools
        return ModelReply(content="ok")


class _ReasoningModel:
    model = "reasoning-1"

    async def complete(self, messages, tools):
        del messages, tools
        # A truncated reasoning-model reply: no visible content, the whole budget
        # spent on reasoning, cut off at the token limit ("length").
        return ModelReply(
            content="",
            finish_reason="length",
            usage={
                "prompt_tokens": 40,
                "completion_tokens": 8096,
                "total_tokens": 8136,
                "reasoning_tokens": 8096,
            },
        )


class SpanHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_span_is_noop_without_active_trace(self) -> None:
        self.assertIsNone(current_node())
        # Must not raise and must still yield a usable scope.
        with span("plan.build") as scope:
            scope.output = {"ignored": True}

    async def test_sequential_spans_are_siblings_and_record_output(self) -> None:
        rec = RecordingTracer()
        trace = rec.start_trace("turn", session_id="s")
        token = bind(trace)
        try:
            with span("plan.build", input="q") as scope:
                scope.output = {"steps": ["a"]}
            with span("step:telemetry.read") as scope:
                scope.output = {"status": "completed"}
        finally:
            unbind(token)

        self.assertEqual([c.name for c in trace.children], ["plan.build", "step:telemetry.read"])
        self.assertTrue(all(c.ended for c in trace.children))
        self.assertEqual(trace.children[0].output, {"steps": ["a"]})

    async def test_span_records_exception_as_error_and_reraises(self) -> None:
        rec = RecordingTracer()
        trace = rec.start_trace("turn", session_id="s")
        token = bind(trace)
        try:
            with self.assertRaises(ValueError):
                with span("boom"):
                    raise ValueError("kaboom")
        finally:
            unbind(token)

        boom = trace.children[-1]
        self.assertEqual(boom.level, "ERROR")
        self.assertIn("ValueError", boom.status_message or "")


class TracedLanguageModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_records_generation_on_current_node(self) -> None:
        rec = RecordingTracer()
        trace = rec.start_trace("turn", session_id="s")
        token = bind(trace)
        try:
            reply = await TracedLanguageModel(_FakeModel()).complete(
                [ChatMessage(role="user", content="hi")], []
            )
        finally:
            unbind(token)

        self.assertEqual(reply.content, "ok")
        self.assertEqual(len(trace.children), 1)
        generation = trace.children[0]
        self.assertEqual(generation.type, "generation")
        self.assertEqual(generation.model, "fake-model-1")
        self.assertEqual(generation.input, [{"role": "user", "content": "hi"}])
        self.assertEqual(generation.output["content"], "ok")
        self.assertTrue(generation.ended)

    async def test_complete_delegates_without_active_trace(self) -> None:
        reply = await TracedLanguageModel(_FakeModel()).complete(
            [ChatMessage(role="user", content="hi")], []
        )
        self.assertEqual(reply.content, "ok")

    async def test_records_finish_reason_and_normalized_usage(self) -> None:
        # A real adapter reports finish_reason + token usage; the generation node
        # must surface both so "deliberate stop vs truncation" and a reasoning
        # model's thinking budget are readable without re-running the call.
        rec = RecordingTracer()
        trace = rec.start_trace("turn", session_id="s")
        token = bind(trace)
        try:
            await TracedLanguageModel(_ReasoningModel()).complete(
                [ChatMessage(role="user", content="hi")], []
            )
        finally:
            unbind(token)

        generation = trace.children[0]
        self.assertEqual(generation.output["finish_reason"], "length")
        # Normalized onto Langfuse usage_details keys, reasoning budget exposed.
        self.assertEqual(
            generation.usage,
            {"input": 40, "output": 8096, "total": 8136, "reasoning": 8096},
        )

    async def test_reply_without_finish_reason_omits_the_key(self) -> None:
        # Doubles/demo models don't report it; the node must not fabricate one.
        rec = RecordingTracer()
        trace = rec.start_trace("turn", session_id="s")
        token = bind(trace)
        try:
            await TracedLanguageModel(_FakeModel()).complete(
                [ChatMessage(role="user", content="hi")], []
            )
        finally:
            unbind(token)

        generation = trace.children[0]
        self.assertNotIn("finish_reason", generation.output)
        self.assertIsNone(generation.usage)

    async def test_records_offered_tools_in_metadata(self) -> None:
        # Tools travel as a separate request parameter, not in the messages, so
        # the node must record them in metadata — otherwise the trace can't show
        # what the model was actually allowed to pick from.
        rec = RecordingTracer()
        trace = rec.start_trace("turn", session_id="s")
        token = bind(trace)
        try:
            await TracedLanguageModel(_FakeModel()).complete(
                [ChatMessage(role="user", content="hi")],
                [
                    ToolSpec(
                        name="search_manual",
                        description="Search the service manual",
                        input_schema={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    )
                ],
            )
        finally:
            unbind(token)

        generation = trace.children[0]
        # Messages stay a clean chat list; the tool schema lives in metadata.
        self.assertEqual(generation.input, [{"role": "user", "content": "hi"}])
        self.assertEqual(generation.metadata["tool_count"], 1)
        tool = generation.metadata["tools"][0]
        self.assertEqual(tool["name"], "search_manual")
        self.assertEqual(tool["description"], "Search the service manual")
        self.assertIn("query", tool["parameters"]["properties"])


class TurnTracingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostic_turn_produces_trace_with_spans_and_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rec = RecordingTracer()
            app = build_demo_app(Path(directory) / "events.jsonl", tracer=rec)
            response = await app.handle(
                Utterance("E123 故障码应该怎么检查？", "trace-session", "cat-306-demo", "operator-1")
            )

        self.assertEqual(len(rec.traces), 1)
        trace = rec.traces[0]
        self.assertEqual(trace.name, "turn")
        self.assertTrue(trace.ended)
        self.assertEqual(trace.output, response.text)

        names = trace.descendant_names()
        self.assertIn("plan.build", names)
        self.assertIn("step:diagnostic.agent", names)

        # The diagnostic agent calls the model, so a generation nests under it.
        agent_span = next(n for n in trace.iter_descendants() if n.name == "step:diagnostic.agent")
        self.assertTrue(any(child.type == "generation" for child in agent_span.children))

    async def test_retrieval_step_span_exposes_evidence_artifacts(self) -> None:
        # knowledge.search puts its hits in an artifact and leaves content empty,
        # so the step span must surface the artifact — otherwise retrieval looks
        # like an empty no-op in the trace.
        with tempfile.TemporaryDirectory() as directory:
            rec = RecordingTracer()
            app = build_demo_app(Path(directory) / "events.jsonl", tracer=rec)
            await app.handle(
                Utterance("E123 故障码应该怎么检查？", "trace-evidence", "cat-306-demo", "operator-1")
            )

        trace = rec.traces[0]
        search_span = next(
            n for n in trace.iter_descendants() if n.name == "step:knowledge.search"
        )
        self.assertEqual(search_span.output["content"], "")
        artifacts = search_span.output["artifacts"]
        self.assertEqual(artifacts[0]["artifact_type"], "knowledge_evidence")
        # The retrieved manual hits are visible in the span, not just swallowed.
        self.assertTrue(artifacts[0]["data"])
        self.assertIn("E123", json.dumps(artifacts[0]["data"], ensure_ascii=False))

    async def test_diagnostic_step_span_input_carries_upstream_evidence(self) -> None:
        # diagnostic.agent takes no arguments; its real input is the evidence its
        # upstream steps produced. The step span must surface that instead of the
        # bare ``undefined`` input a no-argument step would otherwise show.
        with tempfile.TemporaryDirectory() as directory:
            rec = RecordingTracer()
            app = build_demo_app(Path(directory) / "events.jsonl", tracer=rec)
            await app.handle(
                Utterance("E123 故障码应该怎么检查？", "trace-input", "cat-306-demo", "operator-1")
            )

        trace = rec.traces[0]
        diagnose = next(
            n for n in trace.iter_descendants() if n.name == "step:diagnostic.agent"
        )
        self.assertIsNotNone(diagnose.input)
        evidence = diagnose.input["evidence"]
        self.assertIn("knowledge_evidence", {item["artifact_type"] for item in evidence})
        # The upstream manual hit reaches the agent's recorded input, not a no-op.
        self.assertIn("E123", json.dumps(evidence, ensure_ascii=False))


class TracerConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_env_without_keys_selects_noop_tracer(self) -> None:
        from cat_assistant.application.tracing import NoOpTracer

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(_tracer_from_env(), NoOpTracer)

    async def test_langfuse_tracer_requires_optional_package(self) -> None:
        from cat_assistant.adapters.tracing import LangfuseTracer

        try:
            import langfuse  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(RuntimeError, "langfuse"):
                LangfuseTracer(public_key="pk", secret_key="sk")
        else:  # pragma: no cover - depends on optional install
            self.skipTest("langfuse is installed; missing-package path not exercised")


class _FakeObservation:
    """Mimics a langfuse v4 span/generation for adapter-mapping tests."""

    def __init__(self, name: str, as_type: str, **kwargs) -> None:
        self.name = name
        self.as_type = as_type
        self.kwargs = kwargs
        self.children: list["_FakeObservation"] = []
        self.updated: dict | None = None
        self.ended = False

    def start_observation(self, *, name, as_type, input=None, metadata=None, model=None):
        child = _FakeObservation(name, as_type, input=input, metadata=metadata, model=model)
        self.children.append(child)
        return child

    def update(self, **kwargs) -> None:
        self.updated = kwargs

    def end(self, **kwargs) -> None:
        self.ended = True


class _FakeLangfuseClient:
    def __init__(self) -> None:
        self.roots: list[_FakeObservation] = []

    def start_observation(self, *, name, as_type, input=None, metadata=None, model=None):
        root = _FakeObservation(name, as_type, input=input, metadata=metadata)
        self.roots.append(root)
        return root


class LangfuseAdapterMappingTests(unittest.IsolatedAsyncioTestCase):
    """The adapter must speak the v4 start_observation / update / end API."""

    async def test_maps_tree_onto_v4_observation_api(self) -> None:
        from cat_assistant.adapters.tracing import _LangfuseNode, _start_child

        client = _FakeLangfuseClient()
        root = _LangfuseNode(
            _start_child(client, name="turn", as_type="span", input="hi", metadata={"session_id": "s1"})
        )
        step = root.span("step:telemetry.read", metadata={"step_id": "observe"})
        step.end(output={"status": "completed"})
        gen = root.generation("llm.generation", model="qwen", input=[{"role": "user"}])
        gen.end(output={"content": "ok"}, usage={"input": 3, "output": 2})
        root.end(output="done")

        self.assertEqual(len(client.roots), 1)
        root_obs = client.roots[0]
        self.assertEqual(root_obs.as_type, "span")
        self.assertEqual(root_obs.kwargs["metadata"]["session_id"], "s1")

        kinds = {child.name: child.as_type for child in root_obs.children}
        self.assertEqual(kinds["step:telemetry.read"], "span")
        self.assertEqual(kinds["llm.generation"], "generation")

        gen_obs = next(c for c in root_obs.children if c.name == "llm.generation")
        self.assertEqual(gen_obs.kwargs["model"], "qwen")
        # usage maps onto v4's usage_details; output flows through update().
        self.assertEqual(gen_obs.updated["usage_details"], {"input": 3, "output": 2})
        self.assertEqual(gen_obs.updated["output"], {"content": "ok"})
        self.assertTrue(gen_obs.ended and root_obs.ended)


class LoadEnvFileTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_missing_keys_without_overriding_existing(self) -> None:
        from cat_assistant.bootstrap import load_env_file

        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "# a comment\n"
                "export LANGFUSE_PUBLIC_KEY=pk-file\n"
                "LANGFUSE_SECRET_KEY='sk-file'\n"
                "LANGFUSE_HOST=\n",  # empty value is ignored, not set to ""
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-existing"}, clear=True
            ):
                applied = load_env_file(env_path)

                self.assertEqual(applied, env_path)
                # A real environment variable always wins over the file.
                self.assertEqual(os.environ["LANGFUSE_PUBLIC_KEY"], "pk-existing")
                # Missing keys are filled and surrounding quotes stripped.
                self.assertEqual(os.environ["LANGFUSE_SECRET_KEY"], "sk-file")
                # Empty values do not shadow the SDK default.
                self.assertNotIn("LANGFUSE_HOST", os.environ)

    async def test_absent_file_is_a_noop(self) -> None:
        from cat_assistant.bootstrap import load_env_file

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(load_env_file(Path("/nonexistent/does-not-exist.env")))

    async def test_default_search_finds_config_env(self) -> None:
        from cat_assistant.bootstrap import load_env_file

        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "config"
            config_dir.mkdir()
            (config_dir / ".env").write_text(
                "LANGFUSE_SECRET_KEY=sk-from-config\n", encoding="utf-8"
            )
            original_cwd = os.getcwd()
            try:
                os.chdir(directory)  # no ./.env here, only ./config/.env
                with mock.patch.dict(os.environ, {}, clear=True):
                    applied = load_env_file()
                    self.assertEqual(applied, Path("config/.env"))
                    self.assertEqual(os.environ["LANGFUSE_SECRET_KEY"], "sk-from-config")
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
