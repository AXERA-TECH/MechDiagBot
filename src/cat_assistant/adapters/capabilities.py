from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from cat_assistant.application.capabilities import CapabilityExecutionContext
from cat_assistant.application.context import ContextBuilder
from cat_assistant.application.query import DeterministicQueryService
from cat_assistant.application.runner import BoundedAgentRunner
from cat_assistant.domain.capabilities import (
    Artifact,
    CapabilityDescriptor,
    CapabilityResult,
)
from cat_assistant.domain.models import ChatMessage, KnowledgeHit, MachineSnapshot
from cat_assistant.domain.plans import TaskStep
from cat_assistant.domain.ports import KnowledgePort, MachineControlPort


EMPTY_INPUT = {"type": "object", "properties": {}, "additionalProperties": False}
ARTIFACT_OUTPUT = {"type": "object"}

_FAULT_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z]\d{2,5})(?![A-Za-z0-9])", re.IGNORECASE
)
_CURRENT_FOCUS_MARKERS = (
    "当前",
    "目前",
    "现在",
    "本机",
    "这台",
    "本设备",
    "现状",
    "什么问题",
    "哪些问题",
    "存在哪些",
    "这个故障",
    "该故障",
    "这个问题",
    "该问题",
    "这个码",
    "该码",
    "这个异常",
    "该异常",
    "这个报警",
    "该报警",
    "需要如何",
)


def _fault_codes_in(text: str) -> set[str]:
    return {match.group(1).casefold() for match in _FAULT_CODE_RE.finditer(text)}


def _scope_current_fault_hits(
    hits: Sequence[KnowledgeHit],
    machine: MachineSnapshot,
    query: str,
) -> tuple[KnowledgeHit, ...]:
    """Keep a current-fault lookup focused on the codes the machine reports.

    Same-model manuals for other fault codes rank highly in the demo index and
    otherwise make the model discuss (and cite) a fault that is not present.
    Only current-status questions are scoped; asking directly about another code
    still retrieves that code's manual.
    """
    active = {code.casefold() for code in machine.fault_codes}
    if not active or not any(marker in query for marker in _CURRENT_FOCUS_MARKERS):
        return tuple(hits)
    focus = active | _fault_codes_in(query)
    kept: list[KnowledgeHit] = []
    generic: list[KnowledgeHit] = []
    for hit in hits:
        codes = _fault_codes_in(f"{hit.content} {hit.source}")
        if not codes:
            generic.append(hit)
            kept.append(hit)
        elif codes & focus:
            kept.append(hit)
    return tuple(kept or generic or hits[:1])


class TelemetryReadProvider:
    descriptor = CapabilityDescriptor(
        name="telemetry.read",
        version="1.0.0",
        description="读取本轮设备的权威遥测快照（型号、运行状态、故障码、油量、工时等）。",
        input_schema=EMPTY_INPUT,
        output_schema=ARTIFACT_OUTPUT,
        kind="observe",
        source="telemetry",
        produces=("machine_snapshot",),
        # 纯管道：设备快照已随执行上下文提供给每个能力，模型无需专门“读一次”，
        # 故对模型隐藏；规则规划器与依赖图仍照常使用它。
        model_selectable=False,
    )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del arguments, step
        snapshot = await context.current_machine()
        data = {
            "machine_id": snapshot.machine_id,
            "model": snapshot.model,
            "serial_number": snapshot.serial_number,
            "state": snapshot.operating_state,
            "engine_running": snapshot.engine_running,
            "hour_meter": snapshot.hour_meter,
            "next_service_hours": snapshot.next_service_hours,
            "fuel_percent": snapshot.fuel_percent,
            "fault_codes": list(snapshot.fault_codes),
            "captured_at": snapshot.captured_at.isoformat(),
        }
        return CapabilityResult(
            artifacts=(
                Artifact(
                    "machine_snapshot",
                    data,
                    self.descriptor.name,
                    provenance={"authority": "machine_telemetry", "machine_id": snapshot.machine_id},
                    observed_at=snapshot.captured_at,
                ),
            )
        )


class TelemetrySummaryProvider:
    descriptor = CapabilityDescriptor(
        name="telemetry.summarize",
        version="1.0.0",
        description="根据当前遥测，生成一句面向操作员的设备状态摘要。",
        input_schema=EMPTY_INPUT,
        output_schema=ARTIFACT_OUTPUT,
        kind="infer",
        source="telemetry",
        consumes=("machine_snapshot",),
        produces=("telemetry_summary",),
        visible_in_response=True,
    )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del arguments, step
        machine = await context.current_machine()
        faults = "、".join(machine.fault_codes) if machine.fault_codes else "无活动故障码"
        text = (
            f"{machine.model} 当前状态：{machine.operating_state}，"
            f"发动机{'运行中' if machine.engine_running else '已停止'}，"
            f"燃油 {machine.fuel_percent:.0f}%，故障码：{faults}。"
        )
        return CapabilityResult(
            content=text,
            artifacts=(Artifact("telemetry_summary", {"text": text}, self.descriptor.name),),
        )


class UserReportedObservationProvider:
    descriptor = CapabilityDescriptor(
        name="observation.user_report",
        version="1.0.0",
        description="把操作员口述的可见/可听异常（指示灯、报警声、异响等）记录为非权威证据。",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        output_schema=ARTIFACT_OUTPUT,
        kind="observe",
        source="operator",
        trigger_terms=("指示灯", "报警灯", "灯亮", "报警声", "蜂鸣", "异响", "warning light", "alarm"),
        auto_select=True,
        produces=("indicator_report", "audio_alarm_report"),
        # 小众能力：LLM 规划模式下几乎不会被单独选中，且诊断工具已能从用户原文里
        # 读到症状描述。对模型隐藏，规则规划器仍通过 trigger_terms 自动发现它。
        model_selectable=False,
    )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del context, step
        text = str(arguments["text"])
        normalized = text.casefold()
        artifacts = []
        if any(term in normalized for term in ("指示灯", "报警灯", "灯亮", "warning light")):
            artifacts.append(
                Artifact(
                    "indicator_report",
                    {"description": text, "authoritative": False},
                    self.descriptor.name,
                    confidence=0.5,
                    provenance={"reported_by": "operator"},
                )
            )
        if any(term in normalized for term in ("报警声", "蜂鸣", "异响", "alarm")):
            artifacts.append(
                Artifact(
                    "audio_alarm_report",
                    {"description": text, "authoritative": False},
                    self.descriptor.name,
                    confidence=0.5,
                    provenance={"reported_by": "operator"},
                )
            )
        return CapabilityResult(artifacts=tuple(artifacts))


@dataclass(slots=True)
class KnowledgeSearchProvider:
    knowledge: KnowledgePort

    descriptor = CapabilityDescriptor(
        name="knowledge.search",
        version="1.0.0",
        description="在与当前机型匹配的本地知识库与手册中检索资料。",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={"type": "array", "items": {"type": "object"}},
        kind="retrieve",
        source="knowledge",
        # Advertise fault/problem intent as auto_select trigger terms so the
        # planner discovers manual retrieval through the same descriptor-driven
        # path as observation providers, rather than depending solely on the
        # planner's hardcoded keyword gate. This closes the gap where a natural
        # fault-status question ("机器存在什么问题吗") uses "问题" instead of
        # "故障" and would otherwise skip retrieval entirely.
        trigger_terms=("问题", "故障", "毛病", "异常", "报警", "隐患"),
        auto_select=True,
        consumes=("machine_snapshot",),
        produces=("knowledge_evidence",),
    )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del step
        machine = await context.current_machine()
        hits = await self.knowledge.search(str(arguments["query"]), machine=machine)
        hits = _scope_current_fault_hits(hits, machine, str(arguments["query"]))
        data = [
            {
                "content": hit.content,
                "source": hit.source,
                "score": hit.score,
                "document_version": hit.document_version,
            }
            for hit in hits
        ]
        return CapabilityResult(
            artifacts=(
                Artifact(
                    "knowledge_evidence",
                    data,
                    self.descriptor.name,
                    confidence=max((hit.score for hit in hits), default=0.0),
                    provenance={"machine_model": machine.model},
                ),
            )
        )


@dataclass(slots=True)
class DeterministicResponseProvider:
    query_service: DeterministicQueryService

    descriptor = CapabilityDescriptor(
        name="response.compose",
        version="1.0.0",
        description="针对简单的设备事实类问题，给出确定性答复。",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        output_schema=ARTIFACT_OUTPUT,
        kind="summarize",
        source="core",
        consumes=("machine_snapshot",),
        produces=("assistant_response",),
        visible_in_response=True,
        # 确定性简答走规则规划器的“简单事实”通道；对模型而言它与 diagnostic.agent
        # 作为收尾工具重叠，暴露反而增加选择歧义，故对模型隐藏。
        model_selectable=False,
    )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del step
        machine = await context.current_machine()
        response = await self.query_service.answer(str(arguments["text"]), machine)
        return CapabilityResult(
            content=response.text,
            response_category=response.category.value,
            artifacts=(Artifact("assistant_response", {"text": response.text}, self.descriptor.name),),
        )


def _format_diagnosis(text: str, source_line: str | None) -> str:
    """Normalise a diagnosis reply: drop model-written 来源 citations, then append
    a single canonical citation when manual evidence was retrieved.

    Small edge models cite unreliably — repeating the “来源：” prefix on every
    line, gluing the citation mid-sentence onto the 建议 text, or naming a manual
    from a different device. Rather than coax the format out of the model, the
    caller builds the citation deterministically from the (already device-scoped)
    retrieved hits and passes it as ``source_line``; here we remove whatever
    citation the model wrote — whether it leads a line or is glued onto the end
    of a sentence — and re-attach that one clean line, so the operator always
    sees a correct, de-duplicated 来源. ``source_line`` is None when the turn did
    no retrieval — then no citation is shown and any stray model citation is
    still removed (never leave an ungrounded 来源 in the reply).
    """

    def _strip_citation(line: str) -> str:
        cuts = [idx for idx in (line.find("来源："), line.find("来源:")) if idx != -1]
        return line[: min(cuts)].rstrip() if cuts else line

    kept: list[str] = []
    for line in text.splitlines():
        stripped = _strip_citation(line)
        # A line that was *only* a citation collapses to empty — drop it rather
        # than leave a blank hole; a genuinely blank separator line is preserved.
        if not stripped and line.strip():
            continue
        kept.append(stripped)
    body = "\n".join(kept).strip()
    if not source_line:
        return body
    return f"{body}\n\n{source_line}" if body else source_line


@dataclass
class DiagnosticAgentProvider:
    runner: BoundedAgentRunner
    context_builder: ContextBuilder
    # Capability-level ceiling for the whole agent run, in seconds. Defaults low
    # for tests/standalone use; bootstrap raises it to the configured turn budget
    # so a slow edge model is not cut off before the operator's model/turn
    # timeouts. The runner still bounds each model call and the loop bounds the
    # turn, so this wrapper is a backstop, not the primary guard.
    timeout_seconds: int = 90

    # Class-level template: static metadata (name/description/model_selectable) is
    # read at the class level (see test_builtin_menu_is_trimmed_...). Each instance
    # rebuilds it in __post_init__ with the configured timeout so the registry's
    # per-capability wait_for tracks configuration instead of a hardcoded ceiling.
    descriptor = CapabilityDescriptor(
        name="diagnostic.agent",
        version="1.0.0",
        description="综合遥测与已收集的证据，给出有界限的故障诊断结论。",
        input_schema=EMPTY_INPUT,
        output_schema=ARTIFACT_OUTPUT,
        kind="infer",
        source="core",
        timeout_seconds=90,
        visible_in_response=True,
        consumes=("machine_snapshot", "knowledge_evidence"),
        produces=("diagnostic_assessment",),
    )

    def __post_init__(self) -> None:
        # Shadow the class template with an instance descriptor carrying the
        # configured timeout. Class-level access still returns the template.
        self.descriptor = replace(
            type(self).descriptor, timeout_seconds=self.timeout_seconds
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del arguments
        machine = await context.current_machine()
        messages = list(
            await self.context_builder.build(
                session_id=context.utterance.session_id,
                user_text=context.utterance.text,
                machine=machine,
                operator_id=context.utterance.operator_id,
            )
        )
        # Evidence must be spliced in BEFORE the final user turn, never appended
        # after it. ContextBuilder always puts the current request last; appending
        # system messages after it yields a ``...user, system, <assistant>`` turn
        # order that is out-of-distribution for ChatML. Small edge models (observed
        # on Qwen3 2B via AXLLM) then echo a fake ``<|im_start|>user`` turn instead
        # of answering, which _strip_role_leakage collapses to an empty reply.
        # Popping the user turn and re-appending it keeps a standard "system context
        # + user question" prompt so the model actually responds.
        user_turn = messages.pop()
        evidence = context.dependency_artifacts(step)
        reported = [
            artifact for artifact in evidence if artifact.artifact_type.endswith("_report")
        ]
        if reported:
            messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "以下为操作员口述的观察，属非权威信息，下结论前需先核实，"
                        "不能据此直接给出高置信度判断："
                        + json.dumps([artifact.data for artifact in reported], ensure_ascii=False)
                    ),
                )
            )
        knowledge = next(
            (artifact for artifact in evidence if artifact.artifact_type == "knowledge_evidence"),
            None,
        )
        # The citation is built deterministically from the retrieved hits (below),
        # not written by the model: small edge models duplicate the "来源：" prefix,
        # run the citation onto the 建议 sentence, or name a manual from another
        # device. None means the turn did no retrieval, so no citation is shown.
        source_line: str | None = None
        if knowledge is not None:
            # Bound the spliced evidence so a small-context edge model (~2K
            # tokens) cannot overflow: keep only the top few ranked hits (best
            # first from knowledge.search), expose just the content + source the
            # answer needs, and cap each snippet. score/document_version are
            # internal ranking fields — noise for the model and extra tokens.
            hits = knowledge.data if isinstance(knowledge.data, list) else []
            bounded = [
                {
                    "content": str(hit.get("content", ""))[:500],
                    "source": hit.get("source", ""),
                }
                for hit in hits[:3]
                if isinstance(hit, dict)
            ]
            payload = json.dumps(bounded, ensure_ascii=False)
            messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "以下为检索到的手册证据：请据此作答，不要声称做过未执行的检索；"
                        "无需自行书写来源，系统会据此自动附注。\n"
                        + payload
                    ),
                )
            )
            # De-duplicated, order-preserving citation built from the (already
            # device-scoped) sources actually shown to the model.
            sources: list[str] = []
            for hit in bounded:
                src = hit["source"]
                if src and src not in sources:
                    sources.append(src)
            source_line = "来源：" + "；".join(sources) if sources else "来源：未检索到"
        messages.append(user_turn)
        response = await self.runner.run(
            session_id=context.utterance.session_id,
            user_text=context.utterance.text,
            machine=machine,
            operator_id=context.utterance.operator_id,
            initial_messages=messages,
        )
        text = _format_diagnosis(response.text, source_line)
        return CapabilityResult(
            status=(
                "failed"
                if response.category.value == "error"
                else "completed"
            ),
            content=text,
            response_category=response.category.value,
            artifacts=(
                Artifact(
                    "diagnostic_assessment",
                    {"text": text, "evidence_ids": [item.artifact_id for item in evidence]},
                    self.descriptor.name,
                    provenance={"bounded": True},
                ),
            ),
            metadata=response.metadata,
        )


class ActionProposalProvider:
    descriptor = CapabilityDescriptor(
        name="action.propose",
        version="1.0.0",
        description="给出一套需操作员复核的处置/维修步骤建议。",
        input_schema={
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
            "additionalProperties": False,
        },
        output_schema=ARTIFACT_OUTPUT,
        kind="propose",
        source="core",
        visible_in_response=True,
        consumes=("diagnostic_assessment",),
        produces=("action_proposal",),
    )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del step
        machine = await context.current_machine()
        faults = ", ".join(machine.fault_codes) or "未发现活动故障码"
        # 发动机仍在运行时，动手处置前的安全前提通常是先停机熄火（如 E45 冷却系统需
        # 停机冷却后检查）；已停机则只需确认安全工况，不重复要求停机。
        safe_state = (
            "先将设备停机熄火并置于安全工况"
            if machine.engine_running
            else "先确认设备处于安全工况"
        )
        text = (
            f"建议执行方案：{safe_state}（当前：{machine.operating_state}），"
            f"核对故障码 {faults}，相关处置动作须由授权操作员确认后执行。"
        )
        return CapabilityResult(
            content=text,
            artifacts=(Artifact("action_proposal", {"text": text, "request": arguments["request"]}, self.descriptor.name),),
        )


class MachineControlProvider:
    descriptor = CapabilityDescriptor(
        name="machine.control",
        version="1.0.0",
        description="经安全网关审批后，执行改变设备状态的操作。",
        input_schema={
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
            "additionalProperties": False,
        },
        output_schema=ARTIFACT_OUTPUT,
        kind="act",
        risk_level="critical",
        side_effect="device_state_change",
        requires_approval=True,
        source="safety-gateway",
        visible_in_response=True,
        consumes=("action_proposal",),
        produces=("action_result",),
    )

    def __init__(self, executor: MachineControlPort | None = None) -> None:
        self._executor = executor

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del step
        if self._executor is None:
            # Safe default: with no executor bound, a state change cannot be
            # performed. This runs only after policy + approval, so denying here
            # is the last line of defense rather than the primary gate.
            return CapabilityResult(
                status="denied",
                content="当前没有接入受安全网关保护的设备控制执行器。",
            )
        request = arguments["request"]
        machine = await context.current_machine()
        outcome = await self._executor.apply(request, machine=machine)
        if not outcome.accepted:
            return CapabilityResult(status="denied", content=outcome.summary)
        context.update_machine(outcome.machine)
        data: dict[str, Any] = {"request": request, "summary": outcome.summary}
        if outcome.machine is not None:
            data["operating_state"] = outcome.machine.operating_state
            data["engine_running"] = outcome.machine.engine_running
        return CapabilityResult(
            status="completed",
            content=outcome.summary,
            artifacts=(Artifact("action_result", data, self.descriptor.name),),
        )


class VerificationProvider:
    """Re-read telemetry after actuation and compare it with the action result.

    Verification is a first-class pipeline step (``kind="verify"``) rather than
    prose appended to the actuator: it runs only after the approved action, uses
    a live telemetry read, and can therefore detect the difference between "the
    gateway accepted the command" and "the machine actually changed state".
    """

    descriptor = CapabilityDescriptor(
        name="verify.state",
        version="1.0.0",
        description="动作执行后重新读取遥测，复核设备状态是否达到动作预期。",
        input_schema=EMPTY_INPUT,
        output_schema=ARTIFACT_OUTPUT,
        kind="verify",
        source="telemetry",
        consumes=("action_result",),
        produces=("verification_report",),
        visible_in_response=True,
        model_selectable=False,
    )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: CapabilityExecutionContext,
        step: TaskStep,
    ) -> CapabilityResult:
        del arguments
        live_read = await context.refresh_machine()
        machine = context.machine
        evidence = [
            artifact
            for artifact in context.dependency_artifacts(step)
            if artifact.artifact_type == "action_result"
        ]
        if not evidence:
            # Some plans attach verification to the proposal or to a sibling
            # branch; any action result from this plan is still the right reference.
            evidence = list(context.artifacts_of_type("action_result"))
        expected_state: str | None = None
        expected_running: bool | None = None
        for artifact in evidence:
            data = artifact.data if isinstance(artifact.data, dict) else {}
            if isinstance(data.get("operating_state"), str):
                expected_state = data["operating_state"]
            if isinstance(data.get("engine_running"), bool):
                expected_running = data["engine_running"]
        comparisons = expected_running is not None or expected_state is not None
        matched = True
        checks: list[str] = []
        if not live_read:
            checks.append("无法读取动作后的实时遥测，当前显示最后一次已知状态")
        if expected_running is not None:
            agrees = machine.engine_running is expected_running
            matched = matched and agrees
            checks.append(
                f"发动机{'运行中' if machine.engine_running else '已停止'}，"
                f"预期{'运行' if expected_running else '停止'}，"
                + ("一致" if agrees else "不一致")
            )
        if expected_state is not None:
            agrees = machine.operating_state == expected_state
            matched = matched and agrees
            checks.append(
                f"运行状态「{machine.operating_state}」，预期「{expected_state}」，"
                + ("一致" if agrees else "不一致")
            )
        if not comparisons:
            checks.append("动作未声明可核对的遥测字段，以下为执行后的读数")
        faults = "、".join(machine.fault_codes) if machine.fault_codes else "无"
        checks.append(f"故障码：{faults}")
        checks.append(f"观测时间：{machine.captured_at.isoformat()}")
        if not live_read:
            verdict = "复核结果：无法读取动作后的实时遥测，请人工确认设备状态。"
        elif not comparisons:
            verdict = "复核结果：已完成动作后的状态读取（该动作未声明可核对的字段）。"
        elif matched:
            verdict = "复核结果：动作已生效。"
        else:
            verdict = "复核结果：实时读数与动作预期不一致，请人工检查设备。"
        text = verdict + "；".join(checks) + "。"
        return CapabilityResult(
            content=text,
            artifacts=(
                Artifact(
                    "verification_report",
                    {
                        "matched": matched if live_read and comparisons else None,
                        "live_read": live_read,
                        "expected": {
                            "operating_state": expected_state,
                            "engine_running": expected_running,
                        },
                        "observed": {
                            "operating_state": machine.operating_state,
                            "engine_running": machine.engine_running,
                            "fault_codes": list(machine.fault_codes),
                            "captured_at": machine.captured_at.isoformat(),
                        },
                        "text": text,
                    },
                    self.descriptor.name,
                    provenance={"authority": "machine_telemetry"},
                ),
            ),
            metadata={"matched": matched if live_read and comparisons else None},
        )


def builtin_capability_providers(
    *,
    knowledge: KnowledgePort,
    query_service: DeterministicQueryService,
    agent_runner: BoundedAgentRunner,
    context_builder: ContextBuilder,
    machine_control: MachineControlPort | None = None,
    diagnostic_timeout_seconds: int = 90,
) -> tuple[object, ...]:
    return (
        TelemetryReadProvider(),
        TelemetrySummaryProvider(),
        UserReportedObservationProvider(),
        KnowledgeSearchProvider(knowledge),
        DeterministicResponseProvider(query_service),
        DiagnosticAgentProvider(
            agent_runner, context_builder, timeout_seconds=diagnostic_timeout_seconds
        ),
        ActionProposalProvider(),
        MachineControlProvider(machine_control),
        VerificationProvider(),
    )
