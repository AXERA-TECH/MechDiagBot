"""Build bounded, provenance-aware model context for one turn."""

from __future__ import annotations

from collections.abc import Sequence

from cat_assistant.domain.models import ChatMessage, MachineSnapshot, MemoryHit
from cat_assistant.domain.ports import MemoryPort, SessionStorePort


DIAGNOSTIC_SYSTEM_PROMPT = (
    "你是只读设备诊断助手。以当前遥测为准，其余信息仅作参考；保留并引用检索结果的原始来源；"
    "不得声称已控制或修改设备。答复用中文，简洁优先，通常控制在 150 字以内"
    "（用户明确要求详细步骤时除外）。"
    "属于故障、诊断或维修类问题时，按“结论、依据、建议（如有）”组织：每段分别以"
    "“结论：”“依据：”“建议：”开头并另起一行；结论一句话，依据最多 2 条，建议最多 2 条。"
    "依据和建议只写与当前活动故障码或当前问题直接相关的内容；不要讨论遥测中没有出现的"
    "故障码；不要复述已经展示的遥测；不要写“以上为建议”“我无法控制设备”等套话。"
    "普通事实或简单问答直接作答，无需套用该结构。"
    "不要使用 Markdown 记号（* # ` 等）；不要自行书写“来源”行，来源由系统统一附注。"
)

# Used when the console has no device attached. The assistant may answer general
# knowledge questions (principles, maintenance practice, diagnostic reasoning)
# but must not pretend to have telemetry or the ability to touch hardware.
GENERAL_SYSTEM_PROMPT = (
    "你是工业设备诊断与运维助手。当前未接入任何设备：无法读取实时遥测、故障码或设备动作结果，"
    "也不能控制设备。可以回答通用设备知识、原理、保养规范与诊断思路；涉及具体设备状态、"
    "读数、故障码或操作时，明确提示操作员先从设备列表接入设备，不得编造实时数据。"
    "答复用中文，简洁优先；先直接回答，除非用户要求，不展开背景。"
    "不要使用 Markdown 记号（* # ` 等）；不要自行书写“来源”行。"
)


def machine_facts(machine: MachineSnapshot) -> str:
    """One authoritative, all-Chinese device-state line for the model context.

    Shared by ContextBuilder and the BoundedAgentRunner fallback so the two
    never drift, and rendered in Chinese with a clean ``故障码=E123`` list
    (not Python's ``['E123']`` repr) to cut cross-language noise for the edge
    model.
    """
    faults = "、".join(machine.fault_codes) if machine.fault_codes else "无"
    return (
        f"当前设备(权威)：型号={machine.model}；序列号={machine.serial_number}；"
        f"运行状态={machine.operating_state}；故障码={faults}；"
        f"采集时间={machine.captured_at.isoformat()}"
    )


class ContextBuilder:
    """Compose current facts, short-term history and scoped memory.

    Device facts are always fetched by ``AgentLoop`` and passed in here. Memory
    is explicitly labelled as non-authoritative so it cannot replace telemetry.
    The character budget is a dependency-free approximation of a token budget.
    """

    def __init__(
        self,
        sessions: SessionStorePort,
        memory: MemoryPort,
        *,
        history_limit: int = 8,
        max_chars: int = 12_000,
    ) -> None:
        if history_limit < 0 or max_chars < 1:
            raise ValueError("history_limit must be non-negative and max_chars must be positive")
        self._sessions = sessions
        self._memory = memory
        self._history_limit = history_limit
        self._max_chars = max_chars

    async def build(
        self,
        *,
        session_id: str,
        user_text: str,
        machine: MachineSnapshot,
        operator_id: str | None = None,
    ) -> tuple[ChatMessage, ...]:
        try:
            history = await self._sessions.recent_turns(
                session_id,
                limit=self._history_limit,
            )
        except Exception:
            history = ()
        history = tuple(
            turn
            for turn in history
            if turn.machine_id == machine.machine_id
            and (operator_id is None or turn.operator_id in (None, operator_id))
        )
        try:
            summary = await self._sessions.get_summary(session_id)
        except Exception:
            summary = None
        try:
            memories = await self._memory.search(
                user_text,
                machine_id=machine.machine_id,
                operator_id=operator_id,
                limit=5,
            )
        except Exception:
            memories = ()

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=DIAGNOSTIC_SYSTEM_PROMPT),
            ChatMessage(role="system", content=machine_facts(machine)),
        ]
        if summary:
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"会话摘要(非权威): {summary}",
                )
            )
        if memories:
            messages.append(ChatMessage(role="system", content=self._format_memories(memories)))

        for turn in history:
            messages.append(ChatMessage(role="user", content=turn.user_text))
            messages.append(ChatMessage(role="assistant", content=turn.assistant_text))
        messages.append(ChatMessage(role="user", content=user_text))
        return self._fit_budget(messages)

    async def build_general(
        self,
        *,
        session_id: str,
        user_text: str,
        operator_id: str | None = None,
    ) -> tuple[ChatMessage, ...]:
        """Build context for a turn when no device is attached.

        Only turns from the general (machine-less) lane are replayed, and memory
        is scoped to general entries. The device diagnostic prompt and telemetry
        facts are deliberately absent so the model cannot imply it has readings.
        """
        try:
            history = await self._sessions.recent_turns(
                session_id,
                limit=self._history_limit,
            )
        except Exception:
            history = ()
        history = tuple(
            turn
            for turn in history
            if not turn.machine_id
            and (operator_id is None or turn.operator_id in (None, operator_id))
        )
        try:
            summary = await self._sessions.get_summary(session_id)
        except Exception:
            summary = None
        try:
            memories = await self._memory.search(
                user_text,
                machine_id="",
                operator_id=operator_id,
                limit=5,
            )
        except Exception:
            memories = ()

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=GENERAL_SYSTEM_PROMPT),
        ]
        if summary:
            messages.append(
                ChatMessage(role="system", content=f"会话摘要(非权威): {summary}")
            )
        if memories:
            messages.append(ChatMessage(role="system", content=self._format_memories(memories)))

        for turn in history:
            messages.append(ChatMessage(role="user", content=turn.user_text))
            messages.append(ChatMessage(role="assistant", content=turn.assistant_text))
        messages.append(ChatMessage(role="user", content=user_text))
        return self._fit_budget(messages, protected_count=1)

    @staticmethod
    def _format_memories(memories: Sequence[MemoryHit]) -> str:
        lines = [
            "相关记忆(非权威，使用前核验):"
        ]
        lines.extend(
            f"- [{memory.kind}/{memory.source}] {memory.content} (score={memory.score:.2f})"
            for memory in memories
        )
        return "\n".join(lines)

    def _fit_budget(
        self,
        messages: list[ChatMessage],
        *,
        protected_count: int = 2,
    ) -> tuple[ChatMessage, ...]:
        total = sum(len(message.content) for message in messages)
        if total <= self._max_chars:
            return tuple(messages)

        # Preserve system policy/current facts and the current request. Drop the
        # oldest conversational pairs first, then trim memory/summary if needed.
        protected = messages[:protected_count]
        tail = messages[protected_count:]
        while tail and sum(len(item.content) for item in protected + tail) > self._max_chars:
            if len(tail) >= 2 and tail[0].role == "user" and tail[1].role == "assistant":
                tail = tail[2:]
            else:
                tail = tail[1:]
        if sum(len(item.content) for item in protected + tail) <= self._max_chars:
            return tuple(protected + tail)

        current = messages[-1]
        available = max(1, self._max_chars - sum(len(item.content) for item in protected) - 1)
        return tuple(protected + [ChatMessage(role=current.role, content=current.content[:available])])
