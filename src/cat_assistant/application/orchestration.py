"""Task planning, policy evaluation and dependency-graph execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Protocol

from cat_assistant.application import tracing
from cat_assistant.application.approvals import (
    ConfirmationKind,
    PendingApproval,
    is_standalone_cancel,
    is_standalone_confirmation,
    machine_signature,
    parse_confirmation,
    plan_digest,
)
from cat_assistant.application.capabilities import (
    CapabilityExecutionContext,
    CapabilityRegistry,
)
from cat_assistant.application.events import EventRecorder
from cat_assistant.application.intent import has_positive_action, looks_like_question
from cat_assistant.domain.capabilities import Artifact, CapabilityDescriptor
from cat_assistant.domain.models import (
    AssistantResponse,
    ChatMessage,
    MachineSnapshot,
    ModelReply,
    ResponseCategory,
    ToolSpec,
    Utterance,
)
from cat_assistant.domain.plans import StepResult, TaskPlan, TaskStep
from cat_assistant.domain.ports import LanguageModelPort, MachineTelemetryPort


class TaskPlannerPort(Protocol):
    async def build(self, utterance: Utterance, machine: MachineSnapshot) -> TaskPlan: ...


class PolicyEngine:
    """Deterministic gate for capability risk and approval requirements."""

    def evaluate(
        self,
        descriptor: CapabilityDescriptor | None,
        *,
        approved: bool = False,
    ) -> str:
        if descriptor is None:
            return "denied: unknown capability"
        if descriptor.side_effect != "none" or descriptor.requires_approval:
            if not approved:
                return "approval_required"
        return "allow"


class RuleBasedTaskPlanner:
    """Offline planner with registry-driven observation capability discovery."""

    _knowledge_words = (
        "故障",
        "检查",
        "维修",
        "手册",
        "原因",
        "怎么",
        "解决",
        "查询",
        "manual",
        "diagnos",
        "e123",
    )
    _action_words = (
        "启动",
        "停止",
        "停机",
        "移动",
        "升高",
        "降低",
        "设置",
        "修改",
        "执行",
        "start",
        "stop",
        "set ",
    )
    # Only phrasings that ask for an executable plan trigger action.propose. Pure
    # advisory questions ("怎么解决"/"如何处理") are already answered in full by the
    # diagnosis's own 建议 section, and appending the proposal template there just
    # echoes the question back as a nonsensical "执行请求：<问题>". A real action
    # request still reaches machine.control via _action_words / explicit_action.
    _proposal_words = ("方案", "步骤", "给出执行", "给出方案", "处置步骤")

    def __init__(self, capabilities: CapabilityRegistry) -> None:
        self._capabilities = capabilities

    async def build(self, utterance: Utterance, machine: MachineSnapshot) -> TaskPlan:
        del machine
        text = utterance.text.strip()
        lowered = text.casefold()
        needs_knowledge = any(word in lowered for word in self._knowledge_words)
        # "执行" inside "不要执行" / "确认一下油量" must not arm the actuator:
        # require an action word that is not negated and a non-question turn.
        explicit_action = has_positive_action(lowered, self._action_words)
        if looks_like_question(lowered):
            explicit_action = False
        # A read-only request must not accidentally enter the control path just
        # because it contains wording such as “不要执行任何操作”.
        if any(phrase in lowered for phrase in ("只读", "只读检查", "不要执行", "无需执行", "不执行任何操作")):
            explicit_action = False
        proposal_only = any(word in lowered for word in self._proposal_words)
        discovered = self._capabilities.discover(text)
        discovered_by_kind = {
            kind: tuple(item for item in discovered if item.kind == kind)
            for kind in ("observe", "retrieve", "infer", "propose", "act", "verify")
        }
        needs_diagnostic = needs_knowledge or bool(discovered) or (
            not self._is_simple_fact(lowered) and not explicit_action
        )

        steps: list[TaskStep] = [TaskStep("observe", "telemetry.read")]
        evidence_dependencies = ["observe"]
        for index, descriptor in enumerate(discovered_by_kind["observe"], start=1):
            step_id = f"observe-{index}"
            arguments = self._arguments_for(descriptor, text)
            steps.append(TaskStep(step_id, descriptor.name, arguments, ("observe",)))
            evidence_dependencies.append(step_id)

        # The current-state summary is a machine-relevant preamble, not a reply
        # owed to every message. Surface it only when the request genuinely
        # concerns the equipment (fault/knowledge wording or auto-discovered
        # observations) or precedes an action/proposal. Open-ended and
        # conversational turns still reach the model — which already receives the
        # authoritative snapshot in its context — but no longer get the status
        # line prepended, so unrelated questions stay clean.
        needs_summary = (
            needs_knowledge or bool(discovered) or explicit_action or proposal_only
        )
        if needs_summary:
            steps.append(TaskStep("state", "telemetry.summarize", {}, ("observe",)))
            evidence_dependencies.append("state")
        # ``knowledge.search`` can reach the plan two ways: a keyword hit
        # (needs_knowledge) or its own auto_select trigger_terms surfacing it
        # through discover(). Fold both into a single canonical "retrieve" step
        # so a fault-status question ("机器存在什么问题吗") still searches the
        # manuals, the node is never planned twice, and the debug UI keeps
        # labelling it 检索 (the label is keyed on this exact step id).
        knowledge_search_discovered = any(
            descriptor.name == "knowledge.search"
            for descriptor in discovered_by_kind["retrieve"]
        )
        if needs_knowledge or knowledge_search_discovered:
            steps.append(
                TaskStep(
                    "retrieve",
                    "knowledge.search",
                    {"query": text},
                    tuple(evidence_dependencies),
                )
            )
            evidence_dependencies.append("retrieve")
        for index, descriptor in enumerate(
            (
                descriptor
                for descriptor in discovered_by_kind["retrieve"]
                if descriptor.name != "knowledge.search"
            ),
            start=1,
        ):
            step_id = f"retrieve-dynamic-{index}"
            steps.append(
                TaskStep(
                    step_id,
                    descriptor.name,
                    self._arguments_for(descriptor, text),
                    tuple(evidence_dependencies),
                )
            )
            evidence_dependencies.append(step_id)
        for index, descriptor in enumerate(discovered_by_kind["infer"], start=1):
            step_id = f"infer-dynamic-{index}"
            steps.append(
                TaskStep(
                    step_id,
                    descriptor.name,
                    self._arguments_for(descriptor, text),
                    tuple(evidence_dependencies),
                )
            )
            evidence_dependencies.append(step_id)
        if needs_diagnostic:
            steps.append(
                TaskStep(
                    "diagnose",
                    "diagnostic.agent",
                    {},
                    tuple(evidence_dependencies),
                )
            )
            final_dependencies = ["diagnose"]
        elif not needs_knowledge and not needs_summary:
            steps.append(TaskStep("answer", "response.compose", {"text": text}, ("observe",)))
            final_dependencies = ["answer"]
        else:
            final_dependencies = list(evidence_dependencies)

        if proposal_only or explicit_action:
            steps.append(
                TaskStep(
                    "propose",
                    "action.propose",
                    {"request": text},
                    tuple(final_dependencies),
                )
            )
            final_dependencies = ["propose"]
        for index, descriptor in enumerate(discovered_by_kind["propose"], start=1):
            step_id = f"propose-dynamic-{index}"
            steps.append(
                TaskStep(
                    step_id,
                    descriptor.name,
                    self._arguments_for(descriptor, text),
                    tuple(final_dependencies),
                )
            )
            final_dependencies = [step_id]

        action_dependencies = list(final_dependencies)
        action_steps: list[str] = []
        for index, descriptor in enumerate(discovered_by_kind["act"], start=1):
            step_id = f"act-dynamic-{index}"
            steps.append(
                TaskStep(
                    step_id,
                    descriptor.name,
                    self._arguments_for(descriptor, text),
                    tuple(action_dependencies),
                )
            )
            action_steps.append(step_id)
            action_dependencies = [step_id]
        if explicit_action and not proposal_only and not action_steps:
            steps.append(
                TaskStep(
                    "act",
                    "machine.control",
                    {"request": text},
                    tuple(final_dependencies),
                )
            )
            action_steps.append("act")
        verify_dependencies = action_steps or final_dependencies
        # Close the loop deterministically: every actuation is followed by a
        # fresh telemetry verification step, even when the model never planned
        # one. The provider is found by kind, so it is registered (and can be
        # replaced) independently of the planner.
        if action_steps and not discovered_by_kind["verify"]:
            verify_provider = self._capabilities.first_enabled_of_kind("verify")
            if verify_provider is not None:
                steps.append(
                    TaskStep("verify", verify_provider.name, {}, tuple(verify_dependencies))
                )
                verify_dependencies = ["verify"]
        for index, descriptor in enumerate(discovered_by_kind["verify"], start=1):
            step_id = f"verify-dynamic-{index}"
            steps.append(
                TaskStep(
                    step_id,
                    descriptor.name,
                    self._arguments_for(descriptor, text),
                    tuple(verify_dependencies),
                )
            )
            verify_dependencies = [step_id]
        return TaskPlan(steps=tuple(steps), reason="registry-driven request decomposition")

    @staticmethod
    def _arguments_for(descriptor: CapabilityDescriptor, text: str) -> dict[str, str]:
        properties = descriptor.input_schema.get("properties", {})
        if "text" in properties:
            return {"text": text}
        if "query" in properties:
            return {"query": text}
        if "request" in properties:
            return {"request": text}
        return {}

    @staticmethod
    def _is_simple_fact(text: str) -> bool:
        return any(
            word in text
            for word in ("状态", "油量", "燃油", "小时数", "保养", "status", "fuel", "service")
        )


def _is_deterministic_answer_plan(plan: TaskPlan) -> bool:
    """True when a plan is answered entirely by the deterministic response
    composer.

    ``response.compose`` is emitted only by the rule planner's simple
    status/fact branch, which is reached exactly when the turn needs no
    retrieval, diagnosis, proposal or actuation — so such a plan is just
    ``telemetry.read -> response.compose`` and nothing that would benefit from
    the model ever co-occurs with it. That makes the presence of a
    ``response.compose`` step a sound, cheap signal that an LLM planning pass
    would add nothing and can be skipped.
    """
    return any(step.capability == "response.compose" for step in plan.steps)


class ToolCallingTaskPlanner:
    """Let an LLM select from the complete enabled capability set."""

    def __init__(
        self,
        model: LanguageModelPort,
        capabilities: CapabilityRegistry,
        fallback: TaskPlannerPort,
        *,
        max_steps: int | None = None,
        model_call_timeout_seconds: float = 60.0,
        events: EventRecorder | None = None,
    ) -> None:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive, or None for no limit")
        if model_call_timeout_seconds <= 0:
            raise ValueError("model_call_timeout_seconds must be positive")
        self._model = model
        self._capabilities = capabilities
        self._fallback = fallback
        self._max_steps = max_steps
        self._model_call_timeout_seconds = model_call_timeout_seconds
        self._events = events

    async def build(self, utterance: Utterance, machine: MachineSnapshot) -> TaskPlan:
        tools = self._capabilities.model_tools
        if not tools:
            return await self._fallback.build(utterance, machine)
        # The rule planner is pure (no model call), so run it first and reuse its
        # plan on the fast paths below before spending a slow (~10 tok/s) model
        # call on a turn the LLM can't improve.
        fallback_plan = await self._fallback.build(utterance, machine)
        # Skip the (slow, edge-constrained) LLM planning call whenever the
        # deterministic rule plan already fully answers the turn and the LLM
        # cannot improve it:
        #  * deterministic_fact — a bare status/fact turn answered by
        #    response.compose (e.g. "下次保养时间").
        #  * explicit_control — an explicit actuation command ("停机"/"启动"/…)
        #    the rule planner already decomposes into the safe
        #    summarize→propose→machine.control chain. The actuation is
        #    approval-gated regardless, and the planner only *selects* tools (it
        #    can't ask for clarification), so an LLM pass adds latency and
        #    misplan risk with no upside. A compound "诊断+停机" turn still uses
        #    the model for the diagnosis language via diagnostic.agent — only the
        #    orchestration is taken from rules ("LLM for language, rules for
        #    orchestration").
        # Diagnosis/knowledge turns match neither and still get full LLM planning.
        if _is_deterministic_answer_plan(fallback_plan):
            fast_reason: str | None = "deterministic_fact"
        elif any(step.capability == "machine.control" for step in fallback_plan.steps):
            fast_reason = "explicit_control"
        else:
            fast_reason = None
        if fast_reason is not None:
            if self._events is not None:
                await self._events.record(
                    "planning/fast_path",
                    utterance.session_id,
                    reason=fast_reason,
                    steps=[step.capability for step in fallback_plan.steps],
                )
            return fallback_plan
        messages = (
            ChatMessage(role="system", content=self._planning_instructions(machine, tools)),
            ChatMessage(role="user", content=utterance.text),
        )
        try:
            reply = await asyncio.wait_for(
                self._model.complete(messages, tools),
                timeout=self._model_call_timeout_seconds,
            )
        except asyncio.TimeoutError:
            if self._events is not None:
                await self._events.record(
                    "planning/model_timeout",
                    utterance.session_id,
                    timeout_seconds=self._model_call_timeout_seconds,
                )
            return await self._fallback.build(utterance, machine)
        except Exception as exc:
            return await self._reject(
                utterance, machine, "model_error", error=type(exc).__name__
            )
        if not reply.tool_calls:
            return await self._reject(utterance, machine, "empty_selection", reply=reply)
        if self._max_steps is not None and len(reply.tool_calls) > self._max_steps:
            return await self._reject(
                utterance,
                machine,
                "too_many_calls",
                reply=reply,
                call_count=len(reply.tool_calls),
            )

        steps: list[TaskStep] = []
        for index, call in enumerate(reply.tool_calls, start=1):
            capability = self._capabilities.capability_for_model_tool(call.name)
            if capability is None:
                return await self._reject(
                    utterance, machine, "unknown_tool", reply=reply, tool=call.name
                )
            # Real models routinely misname a free-text argument (``text`` for a
            # tool whose slot is ``query``) or add an undeclared field. Reroute a
            # misnamed value onto the declared slot and strip forbidden keys, so a
            # fixable argument slip doesn't sink an otherwise-correct plan; a value
            # missing under every synonym still fails validation below.
            arguments, coercions = self._sanitize_arguments(capability, call.arguments)
            if coercions and self._events is not None:
                await self._events.record(
                    "planning/llm_args_coerced",
                    utterance.session_id,
                    tool=call.name,
                    capability=capability,
                    remapped=[{"from": src, "to": dst} for src, dst in coercions],
                )
            try:
                self._capabilities.validate_arguments(capability, arguments)
            except (LookupError, ValueError) as exc:
                return await self._reject(
                    utterance,
                    machine,
                    "invalid_arguments",
                    reply=reply,
                    tool=call.name,
                    capability=capability,
                    error=str(exc),
                )
            # Every later step receives all earlier artifacts. This intentionally
            # serializes the small-capability route and keeps evidence complete.
            dependencies = tuple(step.step_id for step in steps)
            steps.append(
                TaskStep(
                    step_id=f"model-{index}",
                    capability=capability,
                    arguments=arguments,
                    depends_on=dependencies,
                )
            )
        # Mirror the rule planner's closing step: any model-selected actuation is
        # followed by a verification capability so the operator learns whether the
        # action actually took effect. Verification is discovered by kind and is
        # never one of the model's selectable tools.
        if any(
            (descriptor := self._capabilities.descriptor(step.capability)) is not None
            and descriptor.kind == "act"
            for step in steps
        ) and not any(
            (descriptor := self._capabilities.descriptor(step.capability)) is not None
            and descriptor.kind == "verify"
            for step in steps
        ):
            verify_provider = self._capabilities.first_enabled_of_kind("verify")
            if verify_provider is not None:
                selected = [step.capability for step in steps]
                steps.append(
                    TaskStep(
                        step_id=f"model-{len(steps) + 1}",
                        capability=verify_provider.name,
                        arguments={},
                        depends_on=tuple(step.step_id for step in steps),
                    )
                )
                if self._events is not None:
                    await self._events.record(
                        "planning/llm_plan_repaired",
                        utterance.session_id,
                        reason="appended_verification",
                        appended=verify_provider.name,
                        selected_tools=selected,
                    )
        # A plan that only gathers evidence (telemetry read / knowledge search)
        # but selects no capability that speaks to the operator would execute
        # successfully yet yield the empty "任务已完成，但没有可展示的结果" reply.
        # Rather than discard the model's (correct) evidence selection, append a
        # terminal answer-producing capability that runs last and consumes the
        # gathered evidence, so the operator still gets a diagnosis. Only when no
        # such capability exists do we fall back to the deterministic planner,
        # which always ends with a user-facing step.
        if not any(
            (descriptor := self._capabilities.descriptor(step.capability)) is not None
            and descriptor.visible_in_response
            for step in steps
        ):
            terminal = self._terminal_answer_capability()
            if terminal is None:
                return await self._reject(
                    utterance,
                    machine,
                    "no_visible_answer",
                    reply=reply,
                    capabilities=[step.capability for step in steps],
                )
            if self._events is not None:
                await self._events.record(
                    "planning/llm_plan_repaired",
                    utterance.session_id,
                    reason="appended_terminal_answer",
                    appended=terminal,
                    selected_tools=[step.capability for step in steps],
                )
            steps.append(
                TaskStep(
                    step_id=f"model-{len(steps) + 1}",
                    capability=terminal,
                    arguments={},
                    depends_on=tuple(step.step_id for step in steps),
                )
            )
            return TaskPlan(
                steps=tuple(steps),
                reason="llm tool-call capability selection (terminal answer appended)",
            )
        # ``telemetry.summarize`` is operator-visible, so a plan that selects it
        # (optionally with manual retrieval) passes the check above even though it
        # never produces a diagnosis. The model does this on follow-up questions
        # such as "怎么解决这个故障呢": the turn gets a status line and a silent
        # knowledge artifact, but no answer. When the rule planner determined the
        # turn needs a diagnosis, append one unless the model already selected a
        # diagnosis or a proposal. This is the same "LLM for language, rules for
        # orchestration" split used for actuation: evidence selection stays the
        # model's, but the operator-facing conclusion cannot be dropped.
        diagnosis_required = any(
            step.capability == "diagnostic.agent" for step in fallback_plan.steps
        )
        diagnosis_selected = any(
            step.capability == "diagnostic.agent" for step in steps
        )
        proposal_selected = any(
            (descriptor := self._capabilities.descriptor(step.capability)) is not None
            and descriptor.kind == "propose"
            for step in steps
        )
        if (
            diagnosis_required
            and not diagnosis_selected
            and not proposal_selected
            and self._capabilities.is_enabled("diagnostic.agent")
        ):
            selected = [step.capability for step in steps]
            steps.append(
                TaskStep(
                    step_id=f"model-{len(steps) + 1}",
                    capability="diagnostic.agent",
                    arguments={},
                    depends_on=tuple(step.step_id for step in steps),
                )
            )
            if self._events is not None:
                await self._events.record(
                    "planning/llm_plan_repaired",
                    utterance.session_id,
                    reason="appended_diagnosis",
                    appended="diagnostic.agent",
                    selected_tools=selected,
                )
            return TaskPlan(
                steps=tuple(steps),
                reason="llm tool-call capability selection (diagnosis appended)",
            )
        return TaskPlan(
            steps=tuple(steps),
            reason="llm tool-call capability selection",
        )

    async def _reject(
        self,
        utterance: Utterance,
        machine: MachineSnapshot,
        reason: str,
        *,
        reply: ModelReply | None = None,
        **details: object,
    ) -> TaskPlan:
        """Record why the LLM plan was discarded, then fall back to rules.

        Every LLM-plan rejection used to fall back silently, so the events log
        could not distinguish "LLM planned it" from "LLM plan thrown away" — only
        ``plan.reason`` hinted at it. This makes each rejection observable with a
        machine-readable ``reason`` (empty_selection, unknown_tool,
        invalid_arguments, no_visible_answer, too_many_calls, model_error).

        When the model actually replied (every reason except a raw
        ``model_error`` or timeout), its ``finish_reason`` and token ``usage`` are
        attached too, so the log alone separates truncation
        (``finish_reason="length"``) from a model that deliberately stopped after
        an inadequate selection (``"tool_calls"``/``"stop"``) — no trace dive.
        ``selected_tools`` records the model's *whole* capability selection (not
        just the one that failed), so a plan rejected at its first bad tool still
        shows the full intent behind it.
        """
        if self._events is not None:
            payload = dict(details)
            if reply is not None:
                if reply.finish_reason is not None:
                    payload["finish_reason"] = reply.finish_reason
                if reply.usage:
                    payload["usage"] = reply.usage
                if reply.tool_calls:
                    payload["selected_tools"] = [
                        self._capabilities.capability_for_model_tool(call.name)
                        or call.name
                        for call in reply.tool_calls
                    ]
            await self._events.record(
                "planning/llm_plan_rejected",
                utterance.session_id,
                reason=reason,
                **payload,
            )
        return await self._fallback.build(utterance, machine)

    # Free-text argument keys that all mean "the user's text/intent". Real models
    # routinely pick a plausible-but-wrong one (e.g. ``text`` for a tool whose
    # parameter is ``query`` or ``request``), so a value under any of these is a
    # candidate to reroute onto the slot a schema actually declares.
    _ARGUMENT_SYNONYMS = (
        "text", "query", "request", "input", "prompt", "question", "content", "message", "q",
    )

    def _sanitize_arguments(
        self, capability: str, arguments: dict[str, object]
    ) -> tuple[dict[str, object], list[tuple[str, str]]]:
        """Reroute misnamed free-text keys onto the declared slot, then drop
        undeclared keys.

        Two real-model failure modes are handled: (1) a value placed under a
        wrong-but-synonymous key — ``{"text": ...}`` for a tool whose sole
        required string parameter is ``query`` — is rerouted to that parameter
        rather than discarded; (2) an extra hallucinated field under
        ``additionalProperties: false`` is stripped so it can't sink an
        otherwise-correct plan. A required field with no value under any synonym
        still fails validation. Returns the cleaned arguments plus the list of
        ``(from, to)`` remaps applied, so the caller can record the coercion.
        """
        descriptor = self._capabilities.descriptor(capability)
        if descriptor is None:
            return dict(arguments), []
        schema = descriptor.input_schema
        properties = schema.get("properties") or {}
        required = schema.get("required") or ()
        result = dict(arguments)
        coercions: list[tuple[str, str]] = []
        consumed: set[str] = set()
        for prop in required:
            if prop in result:
                continue
            if (properties.get(prop) or {}).get("type") != "string":
                continue  # only reroute free-text slots, never invent structure
            for synonym in self._ARGUMENT_SYNONYMS:
                if synonym == prop or synonym in consumed:
                    continue
                value = result.get(synonym)
                if isinstance(value, str) and value:
                    result[prop] = value
                    coercions.append((synonym, prop))
                    consumed.add(synonym)
                    break
        if schema.get("additionalProperties") is False:
            allowed = set(properties)
            result = {key: value for key, value in result.items() if key in allowed}
        return result, coercions


    def _terminal_answer_capability(self) -> str | None:
        """Choose a terminal capability to append when the model gathered
        evidence but selected nothing the operator can see.

        The candidate must be safe to append unconditionally: enabled,
        operator-visible, free of side effects and approval gates, and callable
        with empty arguments (so we never have to synthesize a value it
        requires). Among those, prefer an inference/diagnosis step over a bare
        summary, and one that consumes more evidence types — which selects
        ``diagnostic.agent`` (it reasons over the telemetry snapshot and any
        manual hits) ahead of ``telemetry.summarize``. Returns None when the
        registry offers no such capability, leaving the caller to fall back to
        the deterministic planner.
        """
        candidates = [
            descriptor
            for descriptor in self._capabilities.enabled_descriptors
            if descriptor.visible_in_response
            and descriptor.kind not in {"verify"}
            and descriptor.side_effect == "none"
            and not descriptor.requires_approval
            and not (descriptor.input_schema.get("required") or ())
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda descriptor: (descriptor.kind == "infer", len(descriptor.consumes)),
            reverse=True,
        )
        return candidates[0].name

    def _planning_instructions(
        self, machine: MachineSnapshot, tools: Sequence[ToolSpec]
    ) -> str:
        # 列出当前真正可选的“产出答复”工具名（与模型收到的哈希名一致），让
        # “必须以出答复的工具收尾”这条规则指向具体选项，而非抽象类别。
        answer_tools = [
            spec.name
            for spec in tools
            if (descriptor := self._capabilities.descriptor(spec.capability)) is not None
            and descriptor.visible_in_response
        ]
        answer_list = "、".join(answer_tools) if answer_tools else "（当前无可用项）"
        budget = (
            f"最多使用 {self._max_steps} 次工具调用；在此限度内，宁可完整也不要遗漏。\n"
            if self._max_steps is not None
            else ""
        )
        faults = "、".join(machine.fault_codes) if machine.fault_codes else "无"
        return (
            "你是工业设备任务规划器，只挑选并排序工具，不直接答复操作员。必须在这一次就按执行"
            "顺序列出完整计划：系统不会回传中间结果，也不会再次调用你，所以即使打算分步推进，"
            "也要一次把从取证到结论的每一步都列全，不要只选一个工具就收尾。\n"
            "设备/故障/维修类问题的标准链条（按序选全，缺一即不完整）：状态摘要→手册检索→诊断"
            "结论；用户询问解决方法、索取方案或要求处理时，末尾再接行动建议。状态摘要与手册检索属中间取证，"
            "不能作为唯一步骤单独出现，其后必须跟上诊断或建议这类面向操作员的结论工具。\n"
            f"规则：1) 非空计划必须以答复工具收尾：{answer_list}；取证工具排在前面。"
            "2) 故障、异常、报警、维修类问题至少同时包含手册检索与诊断，只给其一即为遗漏。"
            "3) 涉及设备状态、故障或处置的问题开头先给状态摘要；纯闲聊或与设备无关的问题"
            "不选任何工具。"
            "4) 优先只读；改变设备状态须用户明确要求并经操作员审批。"
            "5) 严格遵守工具 schema，只按定义填参数，不臆造字段。\n"
            + budget
            + f"设备：{machine.model}/{machine.machine_id}；运行状态={machine.operating_state}；故障码={faults}。"
        )


def _artifact_trace_view(artifact: Artifact) -> dict[str, object]:
    """Compact, JSON-safe view of a step artifact for its trace span.

    Evidence-gathering steps (retrieve/observe) put their results in artifacts
    and leave ``content`` empty, so a span that records only status+content
    makes them look like empty no-ops. ``default=str``/``skipkeys`` guarantee
    serialisation never raises on exotic payloads, so surfacing artifacts can't
    silently drop the whole step output; large payloads collapse to a bounded
    string while small ones keep their structure for readable rendering.
    """
    serialized = json.dumps(
        artifact.data, default=str, ensure_ascii=False, skipkeys=True
    )
    data: object = (
        serialized[:4000] + "…(truncated)"
        if len(serialized) > 4000
        else json.loads(serialized)
    )
    return {
        "artifact_type": artifact.artifact_type,
        "source_capability": artifact.source_capability,
        "confidence": artifact.confidence,
        "data": data,
    }


def _step_input_view(
    step: TaskStep, context: CapabilityExecutionContext
) -> dict[str, object] | None:
    """Trace ``input`` for a step span: declared arguments + upstream evidence.

    Arguments alone under-describe evidence-consuming capabilities such as
    ``diagnostic.agent``, which take no arguments yet run entirely off the
    artifacts their dependencies produced (accumulated into the execution
    context by the time the step runs). Recording only ``arguments`` leaves those
    steps showing a bare ``undefined`` input; folding in ``dependency_artifacts``
    makes the real upstream evidence the step consumed visible on the span.
    """
    view: dict[str, object] = {}
    if step.arguments:
        view["arguments"] = dict(step.arguments)
    evidence = context.dependency_artifacts(step)
    if evidence:
        view["depends_on"] = list(step.depends_on)
        view["evidence"] = [_artifact_trace_view(artifact) for artifact in evidence]
    return view or None


class PlanExecutor:
    """Execute a validated dependency graph through capability providers."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        policy: PolicyEngine,
        events: EventRecorder,
        *,
        telemetry: MachineTelemetryPort | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._events = events
        self._telemetry = telemetry

    async def execute(
        self,
        plan: TaskPlan,
        *,
        utterance: Utterance,
        machine: MachineSnapshot,
        approved: bool = False,
    ) -> AssistantResponse:
        self._validate(plan)
        context = CapabilityExecutionContext(
            utterance, machine, telemetry=self._telemetry
        )
        completed: set[str] = set()
        results: list[StepResult] = []
        pending = list(plan.steps)
        await self._events.record(
            "plan/created",
            utterance.session_id,
            plan_id=plan.plan_id,
            steps=[step.capability for step in plan.steps],
            reason=plan.reason,
        )

        while pending:
            progress = False
            for step in tuple(pending):
                if not set(step.depends_on).issubset(completed):
                    continue
                pending.remove(step)
                progress = True
                result = await self._execute_step(
                    plan,
                    step,
                    context,
                    approved=approved,
                )
                results.append(result)
                context.artifacts[step.step_id] = result.artifacts
                if result.status == "approval_required":
                    return self._response(results, plan, approved=approved)
                await self._events.record(
                    "plan/step_completed",
                    utterance.session_id,
                    plan_id=plan.plan_id,
                    step_id=step.step_id,
                    capability=step.capability,
                    status=result.status,
                    artifact_ids=[artifact.artifact_id for artifact in result.artifacts],
                )
                if result.status not in {"completed", "skipped"}:
                    return self._response(results, plan, approved=approved)
                completed.add(step.step_id)
            if not progress:
                return AssistantResponse(
                    "任务计划存在未满足的步骤依赖，无法安全执行。",
                    ResponseCategory.ERROR,
                    metadata={"reason": "invalid_plan_dependencies", "plan_id": plan.plan_id},
                )
        return self._response(results, plan, approved=approved)

    async def _execute_step(
        self,
        plan: TaskPlan,
        step: TaskStep,
        context: CapabilityExecutionContext,
        *,
        approved: bool,
    ) -> StepResult:
        with tracing.span(
            f"step:{step.capability}",
            input=_step_input_view(step, context),
            metadata={"step_id": step.step_id},
        ) as scope:
            result = await self._run_step(plan, step, context, approved=approved)
            output: dict[str, object] = {
                "status": result.status,
                "content": result.content[:1000],
            }
            if result.artifacts:
                output["artifacts"] = [
                    _artifact_trace_view(artifact) for artifact in result.artifacts
                ]
            scope.output = output
            if result.status in {"failed", "denied"}:
                scope.level = "ERROR"
                scope.status_message = result.content[:200]
            return result

    async def _run_step(
        self,
        plan: TaskPlan,
        step: TaskStep,
        context: CapabilityExecutionContext,
        *,
        approved: bool,
    ) -> StepResult:
        await self._events.record(
            "plan/step_started",
            context.utterance.session_id,
            plan_id=plan.plan_id,
            step_id=step.step_id,
            capability=step.capability,
        )
        descriptor = self._registry.descriptor(step.capability)
        decision = self._policy.evaluate(descriptor, approved=approved)
        if decision == "approval_required":
            result = StepResult(
                step.step_id,
                step.capability,
                "approval_required",
                "该步骤会改变设备状态，需要操作员确认后才能执行。",
            )
            await self._events.record(
                "plan/approval_required",
                context.utterance.session_id,
                plan_id=plan.plan_id,
                step_id=step.step_id,
                capability=step.capability,
            )
            return result
        if decision != "allow":
            return StepResult(step.step_id, step.capability, "denied", decision)
        try:
            capability_result = await self._registry.execute(step, context)
            return StepResult(
                step.step_id,
                step.capability,
                capability_result.status,
                capability_result.content,
                capability_result.artifacts,
                capability_result.response_category,
                capability_result.metadata,
            )
        except Exception as exc:
            return StepResult(
                step.step_id,
                step.capability,
                "failed",
                f"步骤执行失败：{type(exc).__name__}",
            )

    def _validate(self, plan: TaskPlan) -> None:
        ids = [step.step_id for step in plan.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("task plan contains duplicate step ids")
        known = set(ids)
        for step in plan.steps:
            if not set(step.depends_on).issubset(known):
                raise ValueError(f"step {step.step_id} has an unknown dependency")

    def _visible_results(
        self, results: Sequence[StepResult], *, approved: bool
    ) -> list[StepResult]:
        # After operator approval the state summary and proposal were already
        # shown on the approval turn, so the confirmation reply is just the
        # execution report: surface only the actuation (act/verify) output.
        if approved:
            actuation = [
                item
                for item in results
                if item.content
                and (d := self._registry.descriptor(item.capability)) is not None
                and d.kind in {"act", "verify"}
            ]
            if actuation:
                return actuation
        return [
            item
            for item in results
            if item.content
            and (
                item.status == "approval_required"
                or (
                    (d := self._registry.descriptor(item.capability)) is not None
                    and d.visible_in_response
                )
            )
        ]

    def _response(
        self,
        results: Sequence[StepResult],
        plan: TaskPlan,
        *,
        approved: bool = False,
    ) -> AssistantResponse:
        approval = next((item for item in results if item.status == "approval_required"), None)
        failed = next((item for item in results if item.status in {"failed", "denied"}), None)
        visible = "\n\n".join(
            item.content for item in self._visible_results(results, approved=approved)
        )
        metadata = {
            "plan_id": plan.plan_id,
            "steps": [item.step_id for item in results],
            "capabilities": [item.capability for item in results],
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_type": artifact.artifact_type,
                    "source_capability": artifact.source_capability,
                    "confidence": artifact.confidence,
                }
                for item in results
                for artifact in item.artifacts
            ],
        }
        step_metadata = {
            item.step_id: item.metadata for item in results if item.metadata
        }
        if step_metadata:
            metadata["step_metadata"] = step_metadata
        for item in reversed(results):
            if "reason" in item.metadata:
                metadata["reason"] = item.metadata["reason"]
                break
        if approval:
            return AssistantResponse(
                visible or approval.content,
                ResponseCategory.APPROVAL_REQUIRED,
                requires_confirmation=True,
                metadata=metadata,
            )
        act_completed = approved and any(
            item.status == "completed"
            and (descriptor := self._registry.descriptor(item.capability)) is not None
            and descriptor.kind == "act"
            for item in results
        )
        if failed:
            metadata["failed_step"] = failed.step_id
            # An actuation that already ran is not retracted by a later
            # verification failure: report the action outcome plus the warning
            # instead of labelling the whole turn as an error.
            if act_completed:
                return AssistantResponse(
                    visible or "设备动作已执行，但后续复核未完成，请人工检查设备。",
                    ResponseCategory.ACTION_COMPLETED,
                    metadata=metadata,
                )
            return AssistantResponse(failed.content, ResponseCategory.ERROR, metadata=metadata)
        if act_completed:
            return AssistantResponse(
                visible or "设备动作已执行，请查看执行结果。",
                ResponseCategory.ACTION_COMPLETED,
                metadata=metadata,
            )
        category = next(
            (
                ResponseCategory(item.response_category)
                for item in reversed(results)
                if item.response_category is not None
            ),
            ResponseCategory.DIAGNOSTIC,
        )
        return AssistantResponse(
            visible or "任务已完成，但没有可展示的结果。",
            category,
            metadata=metadata,
        )


class TaskOrchestrator:
    """Facade used by AgentLoop; it has no fixed intent-lane branch.

    A side-effecting plan is kept as a :class:`PendingApproval` bound to the
    operator, machine and machine state that produced it.  Only that same
    operator, on that same machine, against an unchanged state, may execute it;
    a cancellation reply clears it, and an unrelated follow-up supersedes it.
    """

    def __init__(
        self,
        planner: TaskPlannerPort,
        executor: PlanExecutor,
        *,
        events: EventRecorder | None = None,
        approval_ttl_seconds: float = 300.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if approval_ttl_seconds <= 0:
            raise ValueError("approval_ttl_seconds must be positive")
        self._planner = planner
        self._executor = executor
        self._events = events
        self._approval_ttl = timedelta(seconds=float(approval_ttl_seconds))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._pending: dict[str, PendingApproval] = {}

    async def handle(self, utterance: Utterance, machine: MachineSnapshot) -> AssistantResponse:
        session_id = utterance.session_id
        pending = self._pending.get(session_id)
        kind = parse_confirmation(utterance.text)
        if pending is not None and pending.is_expired(self._clock()):
            self._pending.pop(session_id, None)
            await self._record(
                "approval/expired",
                session_id,
                plan_id=pending.plan.plan_id,
                plan_digest=pending.plan_digest,
                expires_at=pending.expires_at.isoformat(),
            )
            if kind is ConfirmationKind.CONFIRM:
                return AssistantResponse(
                    "该操作的确认时限已过，未执行任何设备动作。请重新发起请求。",
                    ResponseCategory.CLARIFICATION,
                    metadata={
                        "reason": "approval_expired",
                        "plan_id": pending.plan.plan_id,
                    },
                )
            pending = None
        if (
            pending is None
            and kind is ConfirmationKind.CANCEL
            and is_standalone_cancel(utterance.text)
        ):
            return self._no_pending_cancel()
        if (
            pending is None
            and kind is ConfirmationKind.CONFIRM
            and is_standalone_confirmation(utterance.text)
        ):
            return AssistantResponse(
                "当前没有待确认的操作，设备未执行任何动作。",
                ResponseCategory.INFORMATION,
                metadata={"reason": "no_pending_approval"},
            )
        if pending is not None:
            if kind is ConfirmationKind.CONFIRM:
                return await self._approve(pending, utterance, machine)
            if kind is ConfirmationKind.CANCEL:
                self._pending.pop(session_id, None)
                await self._record(
                    "approval/rejected",
                    session_id,
                    plan_id=pending.plan.plan_id,
                    plan_digest=pending.plan_digest,
                    operator_id=utterance.operator_id,
                    reason="operator_cancel",
                )
                return AssistantResponse(
                    "已取消待确认操作，设备未执行任何动作。",
                    ResponseCategory.INFORMATION,
                    metadata={
                        "reason": "approval_cancelled",
                        "plan_id": pending.plan.plan_id,
                    },
                )
            # A different request replaces the unconfirmed plan. It is never
            # silently kept, so a later "确认" cannot execute a stale action.
            self._pending.pop(session_id, None)
            await self._record(
                "approval/superseded",
                session_id,
                plan_id=pending.plan.plan_id,
                plan_digest=pending.plan_digest,
                operator_id=utterance.operator_id,
                text=utterance.text[:200],
            )
        with tracing.span("plan.build", input=utterance.text) as scope:
            plan = await self._planner.build(utterance, machine)
            scope.output = {
                "reason": plan.reason,
                "steps": [step.capability for step in plan.steps],
            }
        response = await self._executor.execute(plan, utterance=utterance, machine=machine)
        if response.category is ResponseCategory.APPROVAL_REQUIRED:
            now = self._clock()
            approval = PendingApproval(
                plan=plan,
                session_id=session_id,
                operator_id=utterance.operator_id,
                machine_id=machine.machine_id,
                plan_digest=plan_digest(plan),
                machine_signature=machine_signature(machine),
                requested_at=now,
                expires_at=now + self._approval_ttl,
            )
            self._pending[session_id] = approval
            await self._record(
                "approval/requested",
                session_id,
                plan_id=plan.plan_id,
                plan_digest=approval.plan_digest,
                operator_id=utterance.operator_id,
                machine_id=machine.machine_id,
                expires_at=approval.expires_at.isoformat(),
            )
            metadata = dict(response.metadata)
            metadata["approval"] = {
                "plan_id": plan.plan_id,
                "plan_digest": approval.plan_digest,
                "operator_id": utterance.operator_id,
                "machine_id": machine.machine_id,
                "expires_at": approval.expires_at.isoformat(),
            }
            minutes = self._approval_ttl.total_seconds() / 60
            hint = (
                f"请在 {minutes:g} 分钟内由发起人本人确认；"
                "设备状态发生变化后该方案将自动失效。"
            )
            response = replace(
                response,
                text=f"{response.text}\n\n{hint}",
                metadata=metadata,
            )
        else:
            self._pending.pop(session_id, None)
        return response

    async def _approve(
        self,
        pending: PendingApproval,
        utterance: Utterance,
        machine: MachineSnapshot,
    ) -> AssistantResponse:
        session_id = utterance.session_id
        if plan_digest(pending.plan) != pending.plan_digest:
            self._pending.pop(session_id, None)
            await self._record(
                "approval/plan_changed",
                session_id,
                plan_id=pending.plan.plan_id,
                operator_id=utterance.operator_id,
            )
            return AssistantResponse(
                "待确认方案在生成后发生了变化，未执行。请重新发起请求。",
                ResponseCategory.CLARIFICATION,
                metadata={
                    "reason": "approval_plan_changed",
                    "plan_id": pending.plan.plan_id,
                },
            )
        if pending.operator_id != utterance.operator_id:
            await self._record(
                "approval/operator_mismatch",
                session_id,
                plan_id=pending.plan.plan_id,
                expected_operator=pending.operator_id,
                actual_operator=utterance.operator_id,
            )
            return AssistantResponse(
                "该操作只能由发起人本人确认，当前回复不会执行。请让发起人确认，或重新发起请求。",
                ResponseCategory.CLARIFICATION,
                metadata={
                    "reason": "approval_operator_mismatch",
                    "plan_id": pending.plan.plan_id,
                },
            )
        if pending.machine_id != machine.machine_id:
            await self._record(
                "approval/machine_mismatch",
                session_id,
                plan_id=pending.plan.plan_id,
                expected_machine=pending.machine_id,
                actual_machine=machine.machine_id,
            )
            return AssistantResponse(
                f"该操作针对设备 {pending.machine_id}，当前设备是 {machine.machine_id}，未执行。",
                ResponseCategory.CLARIFICATION,
                metadata={
                    "reason": "approval_machine_mismatch",
                    "plan_id": pending.plan.plan_id,
                },
            )
        if machine_signature(machine) != pending.machine_signature:
            self._pending.pop(session_id, None)
            await self._record(
                "approval/state_changed",
                session_id,
                plan_id=pending.plan.plan_id,
                machine_id=machine.machine_id,
                operator_id=utterance.operator_id,
            )
            return AssistantResponse(
                "设备状态在方案生成后已发生变化，原操作已失效，未执行。请重新发起请求。",
                ResponseCategory.CLARIFICATION,
                metadata={
                    "reason": "approval_state_changed",
                    "plan_id": pending.plan.plan_id,
                },
            )
        # Approval is single-use: consume it before execution so a duplicate or
        # concurrent confirmation cannot run the actuation twice.
        self._pending.pop(session_id, None)
        await self._record(
            "approval/confirmed",
            session_id,
            plan_id=pending.plan.plan_id,
            plan_digest=pending.plan_digest,
            operator_id=utterance.operator_id,
            machine_id=machine.machine_id,
        )
        return await self._executor.execute(
            pending.plan,
            utterance=utterance,
            machine=machine,
            approved=True,
        )

    @staticmethod
    def _no_pending_cancel() -> AssistantResponse:
        return AssistantResponse(
            "当前没有待确认的操作，设备未执行任何动作。",
            ResponseCategory.INFORMATION,
            metadata={"reason": "no_pending_approval"},
        )

    async def cancel_pending(
        self, session_id: str, *, reason: str = "session_cleared"
    ) -> bool:
        """Drop a session's unconfirmed plan (e.g. when its history is deleted).

        Without this, clearing a conversation would leave an approval alive that
        a later "确认执行" in the same session could still run.
        """
        pending = self._pending.pop(session_id, None)
        if pending is None:
            return False
        await self._record(
            "approval/rejected",
            session_id,
            plan_id=pending.plan.plan_id,
            plan_digest=pending.plan_digest,
            operator_id=pending.operator_id,
            reason=reason,
        )
        return True

    async def _record(self, event_type: str, session_id: str, **payload: object) -> None:
        if self._events is not None:
            await self._events.record(event_type, session_id, **payload)
