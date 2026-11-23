from __future__ import annotations

import random
import time
import zlib
from dataclasses import replace
from datetime import datetime, timezone

from cat_assistant.domain.models import MachineSnapshot


class InMemoryTelemetry:
    """Development adapter standing in for a read-only CAN/J1939 gateway."""

    def __init__(self, snapshots: dict[str, MachineSnapshot]) -> None:
        self._snapshots = snapshots

    async def get_snapshot(self, machine_id: str) -> MachineSnapshot:
        try:
            return self._snapshots[machine_id]
        except KeyError as exc:
            raise LookupError(f"unknown machine: {machine_id}") from exc

    def list_snapshots(self) -> tuple[MachineSnapshot, ...]:
        """Enumerate every known machine so hosts can offer a device picker."""
        return tuple(self._snapshots.values())

    def apply_snapshot(self, snapshot: MachineSnapshot) -> None:
        """Replace a machine's authoritative snapshot.

        Used by the simulated control executor so an actuation is reflected by
        the next read, mirroring how a real machine's telemetry catches up after
        a state change. Reads remain the single source of truth.
        """
        self._snapshots[snapshot.machine_id] = snapshot


# 演示机队每 10 分钟"换一次班":窗口内故障分布保持不变,到点后重新洗牌。
# 窗口式设计让设备选择器的指示灯与详情页读数在同一时刻永远一致,
# 且无需后台定时任务——角色完全由当前时间窗口确定。
_FLEET_WINDOW_SECONDS = 600

# 每台设备一组可信的活动故障码;设备"报故障"时从自己的池里取样,
# 让故障码本身也随窗口轮换,而不是永远同一条。
_FLEET_FAULT_POOL: dict[str, tuple[str, ...]] = {
    "cat-306-demo": ("E123", "E232", "E441"),
    "cat-320-demo": ("E456", "E779", "E1382"),
    "cat-950gc-demo": ("E45", "F210", "E360"),
    "cat-d6-demo": ("E538", "E96", "E1045"),
}
_FLEET_BUSY_STATES = ("重载作业", "作业中")   # 带故障但仍在作业
_FLEET_IDLE_STATES = ("作业中", "怠速")       # 正常运行

# 演示故障码释义:调试台把对话中出现的这些码渲染为可悬停查看说明的链接。
# 必须覆盖 _FLEET_FAULT_POOL 与 demo_snapshots() 中出现的每一个码(见测试),
# 这样任意时刻可能出现的故障码都有可读释义。
_FLEET_FAULT_CATALOG: dict[str, str] = {
    "E123": "液压压力异常",
    "E232": "液压油温过高",
    "E441": "先导控制压力异常",
    "E456": "冷却液温度过高",
    "E779": "发动机机油压力偏低",
    "E1382": "排气后处理系统(DPF)压差异常",
    "E45": "发动机冷却系统温度信号异常",
    "F210": "燃油系统压力偏低",
    "E360": "液压系统流量异常",
    "E538": "变速箱油温过高",
    "E96": "蓄电池充电电压异常",
    "E1045": "行走系统液压压力异常",
}


def fault_code_catalog() -> dict[str, str]:
    """演示故障码 -> 释义的只读副本,供调试 UI 悬停提示使用。"""
    return dict(_FLEET_FAULT_CATALOG)


class DemoFleetTelemetry(InMemoryTelemetry):
    """让演示机队"活"起来的只读遥测适配器。

    每 ``_FLEET_WINDOW_SECONDS``(10 分钟)重新洗牌一次机队健康状态,
    并始终维持同一副轮廓——4 台设备中恒有 **2 台运行且有活动故障、
    1 台运行且正常、1 台熄火**。角色由当前时间窗口确定,因此:

    * 同一时刻的任意读取(设备列表 / 单机快照)彼此一致;
    * 跨窗口时各设备角色轮换,故障码也随之刷新;
    * 无需后台线程或可变计时状态,进程内可确定复现。

    此外还强制一条下限:**任意时刻整支机队至少有 1 台显示故障(红灯)**。即便
    操作者用控制指令把报故障的设备逐一停机/接管,覆盖层也会补上一台故障机
    (见 :meth:`_guarantee_fault`),保证演示始终有可诊断的对象。

    这是叠加在基础快照读路径上的薄覆盖层,只改写与"健康角色"相关的字段
    (发动机、工况、故障码、采样时间);机型 / 序列号 / 油量 / 工时等设备
    身份字段仍取自基础快照,保持稳定。
    """

    _WINDOW_SECONDS = _FLEET_WINDOW_SECONDS

    def __init__(self, snapshots: dict[str, MachineSnapshot]) -> None:
        super().__init__(snapshots)
        # 被控制指令执行过的设备:此后直接回报其真实(已作动)状态,不再套用
        # 演示覆盖层,以免"执行动作"的效果被下一次模拟读取掩盖。
        self._actuated: set[str] = set()

    def apply_snapshot(self, snapshot: MachineSnapshot) -> None:
        super().apply_snapshot(snapshot)
        self._actuated.add(snapshot.machine_id)

    async def get_snapshot(self, machine_id: str) -> MachineSnapshot:
        base = await super().get_snapshot(machine_id)
        if machine_id in self._actuated:
            return base
        window = self._window()
        return self._project(base, self._roles(window), window)

    def list_snapshots(self) -> tuple[MachineSnapshot, ...]:
        window = self._window()
        roles = self._roles(window)
        return tuple(
            snapshot
            if snapshot.machine_id in self._actuated
            else self._project(snapshot, roles, window)
            for snapshot in super().list_snapshots()
        )

    # -- 时间窗口 → 角色分配 --------------------------------------------------
    def _window(self) -> int:
        return int(time.time() // self._WINDOW_SECONDS)

    def _roles(self, window: int) -> dict[str, str]:
        """把某个 10 分钟窗口确定性地映射为每台设备的健康角色。

        基础分配:2 台故障、1 台正常、1 台熄火。在此之上再强制一条下限——
        整支机队任意时刻至少有 1 台显示故障(红灯),详见 :meth:`_guarantee_fault`。
        """
        order = sorted(self._snapshots)
        random.Random(window).shuffle(order)
        roles: dict[str, str] = {}
        for position, machine_id in enumerate(order):
            if position < 2:
                roles[machine_id] = "fault"      # 运行 + 有故障(红)
            elif position == 2:
                roles[machine_id] = "healthy"    # 运行 + 正常(绿)
            else:
                roles[machine_id] = "stopped"    # 熄火(灰)
        self._guarantee_fault(order, roles)
        return roles

    def _guarantee_fault(self, order: list[str], roles: dict[str, str]) -> None:
        """保证整支机队任意时刻至少有 1 台显示故障(红灯)。

        被控制指令钉住(接管)的设备如实反映真实状态、不套用覆盖层。若操作者把
        原本报故障的设备都停机/接管后,覆盖层里可能已无故障机 —— 此时挑一台
        "未钉住"的设备改判为故障角色补上,避免出现 0 故障的空档。

        仅两种情况不干预:已钉住的设备中已有故障机(运行且带故障码),下限已由
        它满足;或全部设备都被钉住(无可改判的设备),此时应如实反映真实控制结果。
        """
        pinned_faulted = any(
            machine_id in self._actuated
            and self._snapshots[machine_id].engine_running
            and self._snapshots[machine_id].fault_codes
            for machine_id in order
        )
        if pinned_faulted:
            return
        free = [machine_id for machine_id in order if machine_id not in self._actuated]
        if free and not any(roles[machine_id] == "fault" for machine_id in free):
            roles[free[0]] = "fault"

    def _project(
        self, base: MachineSnapshot, roles: dict[str, str], window: int
    ) -> MachineSnapshot:
        role = roles.get(base.machine_id, "healthy")
        captured_at = datetime.now(timezone.utc)
        if role == "stopped":
            return replace(
                base,
                engine_running=False,
                operating_state="熄火",
                fault_codes=(),
                captured_at=captured_at,
            )
        # 每台设备以"窗口 + 设备号"独立播种,故障码 / 工况互不相关,且不依赖
        # 字符串哈希(规避 PYTHONHASHSEED 带来的跨进程不确定性)。
        rng = random.Random(window * 1_000_003 + zlib.crc32(base.machine_id.encode("utf-8")))
        if role == "fault":
            pool = _FLEET_FAULT_POOL.get(base.machine_id) or ("E000",)
            count = rng.randint(1, min(2, len(pool)))
            return replace(
                base,
                engine_running=True,
                operating_state=rng.choice(_FLEET_BUSY_STATES),
                fault_codes=tuple(rng.sample(pool, count)),
                captured_at=captured_at,
            )
        return replace(
            base,
            engine_running=True,
            operating_state=rng.choice(_FLEET_IDLE_STATES),
            fault_codes=(),
            captured_at=captured_at,
        )


def demo_snapshot(machine_id: str = "cat-306-demo") -> MachineSnapshot:
    return MachineSnapshot(
        machine_id=machine_id,
        model="Cat 306 CR",
        serial_number="DEMO-306-001",
        engine_running=True,
        operating_state="怠速",
        hour_meter=1248.2,
        next_service_hours=1250.0,
        fuel_percent=68.0,
        fault_codes=("E123",),
    )


def demo_snapshots() -> dict[str, MachineSnapshot]:
    """A small fleet of demo machines with varied models and health states."""
    fleet = (
        demo_snapshot(),
        MachineSnapshot(
            machine_id="cat-320-demo",
            model="Cat 320",
            serial_number="DEMO-320-014",
            engine_running=True,
            operating_state="作业中",
            hour_meter=5321.0,
            next_service_hours=5400.0,
            fuel_percent=82.0,
            fault_codes=(),
        ),
        MachineSnapshot(
            machine_id="cat-950gc-demo",
            model="Cat 950 GC",
            serial_number="DEMO-950-007",
            engine_running=False,
            operating_state="熄火",
            hour_meter=8123.5,
            next_service_hours=8130.0,
            fuel_percent=14.0,
            fault_codes=("E45", "F210"),
        ),
        MachineSnapshot(
            machine_id="cat-d6-demo",
            model="Cat D6",
            serial_number="DEMO-D6-021",
            engine_running=True,
            operating_state="怠速",
            hour_meter=3102.7,
            next_service_hours=3150.0,
            fuel_percent=47.0,
            fault_codes=(),
        ),
    )
    return {snapshot.machine_id: snapshot for snapshot in fleet}
