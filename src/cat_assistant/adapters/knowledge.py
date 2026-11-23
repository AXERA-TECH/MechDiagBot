from __future__ import annotations

from collections.abc import Sequence
import re

from cat_assistant.domain.models import KnowledgeHit, MachineSnapshot


def _is_other_model_manual(hit: KnowledgeHit, model: str) -> bool:
    """True when a hit is a device manual for a *different* model than ``model``.

    Retrieval must honour the capability's "与当前机型匹配" contract: a Cat 950 GC
    turn should never cite a Cat 306 CR manual. A hit is kept when it names the
    current model, or names no specific device model at all (generic safety /
    operations guidance). Detection is brand-relative: look for the machine's
    brand token (e.g. "Cat") followed by a model designator containing a digit
    ("306", "950", "D6"); such a mention that isn't the current model marks the
    hit as another device's manual. This keeps the demo index dependency-free
    without a structured per-document model field.
    """
    haystack = f"{hit.content} {hit.source}".casefold()
    if model.casefold() in haystack:
        return False
    brand = model.split()[0].casefold() if model.split() else ""
    if not brand:
        return False
    return re.search(rf"\b{re.escape(brand)}\s+\S*\d\S*", haystack) is not None


class InMemoryKnowledge:
    """Development adapter; a local hybrid index can implement the same port."""

    def __init__(self, documents: Sequence[KnowledgeHit]) -> None:
        self._documents = tuple(documents)

    async def search(
        self,
        query: str,
        *,
        machine: MachineSnapshot,
        limit: int = 3,
    ) -> Sequence[KnowledgeHit]:
        # Keep the demo index dependency-free while still making fault-code
        # retrieval meaningful. Machine model/fault codes receive a relevance
        # boost, followed by query token overlap.
        tokens = {token.casefold() for token in re.findall(r"[\w-]+", query) if len(token) > 1}
        required = {machine.model.casefold(), *(code.casefold() for code in machine.fault_codes)}
        ranked = []
        for index, hit in enumerate(self._documents):
            # Scope to the current machine: another device's manual is dropped
            # before ranking (so it can't take a top-``limit`` slot from a
            # relevant hit); model-agnostic docs and this model's docs stay.
            if _is_other_model_manual(hit, machine.model):
                continue
            haystack = f"{hit.content} {hit.source}".casefold()
            # Explicit query terms (especially a fault code) should win over
            # the machine's default-fault boost when an operator asks about a
            # different code.
            score = hit.score + sum(3.0 for token in tokens if token in haystack)
            score += sum(2.0 for token in required if token and token in haystack)
            ranked.append((score, -index, hit))
        ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in ranked[:limit])


def demo_documents() -> tuple[KnowledgeHit, ...]:
    return (
        KnowledgeHit(
            content=(
                "Cat 306 CR 故障码 E123 表示演示用液压压力异常。"
                "检查前应停机、释放液压压力，并由授权技师按手册步骤诊断。"
            ),
            source="Cat 306 CR Service Manual §8.4 (demo)",
            score=0.94,
            document_version="2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 306 CR 故障码 E456 表示冷却液温度过高。降低负载并观察温度；必要时安全停机。冷却后检查冷却液液位、散热器堵塞、风扇皮带和温度传感器。禁止热机立即打开散热器盖。",
            "Cat 306 CR Service Manual §6.2 (demo)", .93, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 306 CR 故障码 E232 表示液压油温过高。停止重载动作并怠速散热；检查液压油位、油冷却器堵塞、风扇和回油滤芯。油温未降前不得继续作业，禁止立即打开高温液压部件。",
            "Cat 306 CR Service Manual §8.6 (demo)", .92, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 306 CR 故障码 E441 表示先导控制压力异常。停机、落下工作装置并释放先导压力；检查先导泵、滤芯、管路泄漏和压力传感器，由授权技师测量先导压力后恢复作业。",
            "Cat 306 CR Service Manual §8.9 (demo)", .91, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 320 故障码 E456 表示冷却液温度过高。降低负载并观察温度；必要时安全停机。冷却后检查冷却液液位、散热器堵塞、风扇皮带和温度传感器。禁止热机立即打开散热器盖。",
            "Cat 320 Service Manual §6.2 (demo)", .93, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 320 故障码 E779 表示发动机机油压力偏低。立即安全停机，检查机油液位、泄漏、机油滤清器和压力传感器；未确认压力正常前不得重新起动或带载运行，以免损坏发动机。",
            "Cat 320 Service Manual §3.4 (demo)", .94, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 320 故障码 E1382 表示排气后处理系统(DPF)压差异常。降低负载并安全停机，检查 DPF 压差传感器及取样管路、积碳和再生条件；禁止在易燃环境强制再生，由授权技师确认后恢复作业。",
            "Cat 320 Service Manual §5.7 (demo)", .93, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 950 GC 故障码 E45 表示发动机冷却系统温度信号异常。停机冷却后检查冷却液、线束和传感器；若温度持续升高，不得继续重载作业。",
            "Cat 950 GC Service Manual §6.1 (demo)", .92, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 950 GC 故障码 F210 表示燃油系统压力偏低。检查燃油液位、燃油滤清器和供油管路是否进气或堵塞；不得在未确认压力正常前反复起动发动机。",
            "Cat 950 GC Service Manual §7.3 (demo)", .91, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 950 GC 故障码 E360 表示液压系统流量异常。停止工作装置动作并释放液压压力；检查液压油位、泵流量、滤芯堵塞和流量传感器，确认流量正常前不得恢复带载作业。",
            "Cat 950 GC Service Manual §8.2 (demo)", .90, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat D6 故障码 E538 表示变速箱油温过高。停止推土负载并置于空挡怠速散热；检查变速箱油位、油冷却器、滤芯和温度传感器。油温未恢复正常前不得继续重载行走。",
            "Cat D6 Service Manual §7.4 (demo)", .92, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat D6 故障码 E96 表示蓄电池充电电压异常。检查蓄电池接线柱、发电机皮带、线束和充电电压；避免依赖蓄电池继续作业，确认充电系统正常后再投入下一班次。",
            "Cat D6 Service Manual §4.8 (demo)", .91, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat D6 故障码 E1045 表示行走系统液压压力异常。停止行走、落下工作装置并释放液压压力；检查行走马达、管路泄漏、滤芯和压力传感器，由授权技师排查后恢复行走。",
            "Cat D6 Service Manual §8.5 (demo)", .91, "2026.1-demo",
        ),
        KnowledgeHit(
            "故障码处理通用安全要求：先确认设备处于安全状态，停止危险动作并记录故障发生时间、负载和环境；涉及液压、燃油或高温系统时由授权技师执行检修。",
            "CAT Safety Handbook §1 (demo)", .80, "2026.1-demo",
        ),
        KnowledgeHit(
            "Cat 306 CR 维护提醒：接近计划小时数时安排停机保养，检查发动机油、液压油、滤芯和润滑点，并记录更换件批次。",
            "Cat 306 CR Maintenance Schedule (demo)", .78, "2026.1-demo",
        ),
    )
