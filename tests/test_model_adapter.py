from __future__ import annotations

import json
import unittest
from unittest import mock

from cat_assistant.adapters.model import OpenAICompatibleModel, _usage_view
from cat_assistant.domain.models import ChatMessage


class UsageViewTests(unittest.TestCase):
    def test_extracts_counts_and_reasoning_tokens(self) -> None:
        view = _usage_view(
            {
                "prompt_tokens": 40,
                "completion_tokens": 8096,
                "total_tokens": 8136,
                "completion_tokens_details": {"reasoning_tokens": 8096},
            }
        )
        self.assertEqual(
            view,
            {
                "prompt_tokens": 40,
                "completion_tokens": 8096,
                "total_tokens": 8136,
                "reasoning_tokens": 8096,
            },
        )

    def test_missing_or_empty_usage_is_none(self) -> None:
        self.assertIsNone(_usage_view(None))
        self.assertIsNone(_usage_view({}))
        self.assertIsNone(_usage_view("not-a-dict"))

    def test_partial_usage_keeps_only_present_counts(self) -> None:
        # No completion_tokens_details at all: reasoning_tokens simply absent.
        self.assertEqual(_usage_view({"prompt_tokens": 10}), {"prompt_tokens": 10})


class OpenAICompatibleParsingTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _model() -> OpenAICompatibleModel:
        return OpenAICompatibleModel(model="m", base_url="http://x/v1", api_key="")

    async def test_complete_surfaces_finish_reason_and_usage(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "cap_x", "arguments": "{}"}}
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 8096,
                "total_tokens": 8136,
                "completion_tokens_details": {"reasoning_tokens": 8096},
            },
        }
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="hi")], [])

        self.assertEqual(reply.finish_reason, "length")
        self.assertEqual(reply.usage["reasoning_tokens"], 8096)
        self.assertEqual(reply.tool_calls[0].name, "cap_x")
        self.assertEqual(reply.tool_calls[0].call_id, "c1")

    async def test_complete_without_usage_leaves_fields_none(self) -> None:
        response = {"choices": [{"message": {"content": "ok"}}]}
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="hi")], [])

        self.assertEqual(reply.content, "ok")
        self.assertIsNone(reply.finish_reason)
        self.assertIsNone(reply.usage)

    async def test_missing_message_still_raises_valueerror(self) -> None:
        # The error contract must survive the choice/message split refactor.
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value={"choices": []}):
            with self.assertRaises(ValueError):
                await model.complete([ChatMessage(role="user", content="hi")], [])


class QwenTagRecoveryTests(unittest.IsolatedAsyncioTestCase):
    """AXLLM/edge servers return Qwen <tool_call>/<think> as plain text."""

    @staticmethod
    def _model(**kwargs) -> OpenAICompatibleModel:
        return OpenAICompatibleModel(model="m", base_url="http://x/v1", api_key="", **kwargs)

    async def test_recovers_tool_call_and_strips_think_from_content(self) -> None:
        content = (
            "<think>先读遥测，再决定下一步。</think>\n"
            '<tool_call>\n{"name": "cap_telemetry_read_ab57478e", "arguments": {}}\n</tool_call>'
        )
        response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="机器有什么问题")], [])

        self.assertEqual(len(reply.tool_calls), 1)
        self.assertEqual(reply.tool_calls[0].name, "cap_telemetry_read_ab57478e")
        self.assertEqual(reply.tool_calls[0].arguments, {})
        # Neither the reasoning nor the raw tag reaches the operator-facing text.
        self.assertNotIn("<think>", reply.content)
        self.assertNotIn("<tool_call>", reply.content)

    async def test_recovers_multiple_calls_with_nested_arguments(self) -> None:
        # The tag body is delimited by </tool_call>, so nested JSON survives.
        content = (
            '<tool_call>{"name": "cap_a", "arguments": {"query": "E123 检修步骤"}}</tool_call>'
            '<tool_call>{"name": "cap_b", "arguments": {}}</tool_call>'
        )
        response = {"choices": [{"message": {"content": content}}]}
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="x")], [])

        self.assertEqual([call.name for call in reply.tool_calls], ["cap_a", "cap_b"])
        self.assertEqual(reply.tool_calls[0].arguments, {"query": "E123 检修步骤"})

    async def test_recovers_axllm_function_keyed_text_calls(self) -> None:
        # AXera AXLLM's Qwen3.5-4B chat template emits the tool NAME under the
        # key "function" (a bare string) rather than "name", and leaves an empty
        # <think></think> block inline. Every block must still be recovered — the
        # planner would otherwise log empty_selection and discard the LLM plan.
        content = (
            "<think>\n\n</think>\n\n"
            '<tool_call>\n{"function": "cap_telemetry_summarize_ba63631c", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"function": "cap_knowledge_search_f2405c61", "arguments": {"query": "E45 F210"}}\n</tool_call>\n'
            '<tool_call>\n{"function": "cap_diagnostic_agent_95bd00a7", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"function": "cap_action_propose_4bd7c084", "arguments": {"request": "E45 F210"}}\n</tool_call>'
        )
        response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="有什么故障")], [])

        self.assertEqual(
            [call.name for call in reply.tool_calls],
            [
                "cap_telemetry_summarize_ba63631c",
                "cap_knowledge_search_f2405c61",
                "cap_diagnostic_agent_95bd00a7",
                "cap_action_propose_4bd7c084",
            ],
        )
        self.assertEqual(reply.tool_calls[1].arguments, {"query": "E45 F210"})
        self.assertEqual(reply.tool_calls[3].arguments, {"request": "E45 F210"})
        self.assertEqual(reply.content, "")

    async def test_structured_tool_call_with_function_string_name(self) -> None:
        # Defensive: if AXLLM ever surfaces these as a structured tool_calls entry
        # but keeps its non-standard shape ("function" holding the name string),
        # the structured path must parse it too rather than raise on function.get.
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "function": "cap_knowledge_search_f2405c61",
                             "arguments": {"query": "E45"}}
                        ],
                    }
                }
            ]
        }
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="x")], [])

        self.assertEqual(reply.tool_calls[0].name, "cap_knowledge_search_f2405c61")
        self.assertEqual(reply.tool_calls[0].arguments, {"query": "E45"})
        self.assertEqual(reply.tool_calls[0].call_id, "c1")

    async def test_normalizes_whitespace_split_into_opaque_hash(self) -> None:
        # The 4B detokenizer splits the non-linguistic hash suffix and injects a
        # space ("cap_..._b a63631c"). The recovered NAME must be whitespace-
        # normalized so it still maps back to a capability rather than being
        # rejected as unknown — while a legitimate space in an ARGUMENT value is
        # preserved untouched.
        content = (
            '<tool_call>{"function": "cap_telemetry_summarize_b a63631c", "arguments": {}}</tool_call>'
            '<tool_call>{"function": "cap_action_propose_4bd7c084", "arguments": {"request": "处理 E45 F210"}}</tool_call>'
        )
        response = {"choices": [{"message": {"content": content}}]}
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="x")], [])

        self.assertEqual(reply.tool_calls[0].name, "cap_telemetry_summarize_ba63631c")
        self.assertEqual(reply.tool_calls[1].name, "cap_action_propose_4bd7c084")
        self.assertEqual(reply.tool_calls[1].arguments, {"request": "处理 E45 F210"})

    async def test_structured_tool_calls_take_precedence_and_think_stripped(self) -> None:
        # When the server DID parse tool_calls, keep them (no double counting) but
        # still strip any inline <think> from the content.
        response = {
            "choices": [
                {
                    "message": {
                        "content": "<think>reasoning</think>",
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "cap_x", "arguments": "{}"}}
                        ],
                    }
                }
            ]
        }
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="x")], [])

        self.assertEqual([call.name for call in reply.tool_calls], ["cap_x"])
        self.assertEqual(reply.content, "")

    async def test_truncated_tool_call_tag_degrades_to_no_calls(self) -> None:
        # A cut-off generation (finish_reason="length") with a half-written tag
        # must not raise; it degrades to no calls so the planner can fall back.
        content = '<think>...</think>\n<tool_call>{"name": "cap_x", "argum'
        response = {"choices": [{"finish_reason": "length", "message": {"content": content}}]}
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="x")], [])

        self.assertEqual(reply.tool_calls, ())
        self.assertEqual(reply.finish_reason, "length")

    async def test_disable_thinking_appends_no_think_to_system_prompt(self) -> None:
        captured: dict[str, object] = {}

        def _capture(request):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return {"choices": [{"message": {"content": "ok"}}]}

        model = self._model(disable_thinking=True)
        with mock.patch.object(OpenAICompatibleModel, "_request", side_effect=_capture):
            await model.complete(
                [
                    ChatMessage(role="system", content="你是规划器。"),
                    ChatMessage(role="user", content="机器有什么问题"),
                ],
                [],
            )

        system = captured["body"]["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertTrue(system["content"].endswith("/no_think"))

    async def test_thinking_enabled_by_default_leaves_prompt_untouched(self) -> None:
        captured: dict[str, object] = {}

        def _capture(request):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return {"choices": [{"message": {"content": "ok"}}]}

        model = self._model()  # disable_thinking defaults to False
        with mock.patch.object(OpenAICompatibleModel, "_request", side_effect=_capture):
            await model.complete([ChatMessage(role="system", content="sys")], [])

        self.assertNotIn("/no_think", captured["body"]["messages"][0]["content"])

    async def test_role_token_leakage_is_truncated_from_content(self) -> None:
        # The server did not stop on <|im_end|>, so the model ran past its turn and
        # began emitting a fake next turn. Only the real answer must survive.
        content = (
            "诊断结论：E123 表示液压压力异常，请先停机泄压。"
            "<|im_end|>\n<|im_start|>user\n机器有什么问题"
        )
        response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="x")], [])

        self.assertEqual(reply.content, "诊断结论：E123 表示液压压力异常，请先停机泄压。")
        self.assertNotIn("<|im_start|>", reply.content)
        self.assertNotIn("<|im_end|>", reply.content)

    async def test_tool_call_inside_leaked_turn_is_not_recovered(self) -> None:
        # A <tool_call> that appears only inside the hallucinated next turn must be
        # dropped with the leaked turn, not parsed as a real selection.
        content = (
            "好的，我来查一下。"
            '<|im_start|>assistant\n<tool_call>{"name": "cap_x", "arguments": {}}</tool_call>'
        )
        response = {"choices": [{"message": {"content": content}}]}
        model = self._model()
        with mock.patch.object(OpenAICompatibleModel, "_request", return_value=response):
            reply = await model.complete([ChatMessage(role="user", content="x")], [])

        self.assertEqual(reply.tool_calls, ())
        self.assertEqual(reply.content, "好的，我来查一下。")

    async def test_stop_sequences_are_sent_in_payload(self) -> None:
        captured: dict[str, object] = {}

        def _capture(request):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return {"choices": [{"message": {"content": "ok"}}]}

        model = self._model(stop=("<|im_end|>", "<|im_start|>"))
        with mock.patch.object(OpenAICompatibleModel, "_request", side_effect=_capture):
            await model.complete([ChatMessage(role="user", content="hi")], [])

        self.assertEqual(captured["body"]["stop"], ["<|im_end|>", "<|im_start|>"])

    async def test_no_stop_configured_omits_the_field(self) -> None:
        captured: dict[str, object] = {}

        def _capture(request):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return {"choices": [{"message": {"content": "ok"}}]}

        model = self._model()  # stop defaults to ()
        with mock.patch.object(OpenAICompatibleModel, "_request", side_effect=_capture):
            await model.complete([ChatMessage(role="user", content="hi")], [])

        self.assertNotIn("stop", captured["body"])


if __name__ == "__main__":
    unittest.main()
