from __future__ import annotations

import dataclasses
from typing import Protocol

from cat_assistant.application.intent import (
    ControlIntent,
    control_command,
    has_negation_cue,
)
from cat_assistant.domain.models import ControlOutcome, MachineSnapshot


class SupportsSnapshotWrite(Protocol):
    def apply_snapshot(self, snapshot: MachineSnapshot) -> None: ...


class SimulatedMachineControl:
    """A virtual Safety Gateway executor for offline demonstrations.

    It closes the propose -> approve -> execute -> verify loop without any real
    actuator: a recognized command mutates the in-memory telemetry snapshot so
    the next read reflects the new state, exactly as a real machine's telemetry
    would catch up after an actuation. It is deliberately the only write path
    and runs only after the deterministic policy gate and operator confirmation.

    The device it acts on is itself simulated (see ``demo_snapshots``); this
    executor simply completes that simulation rather than controlling hardware.
    Negated, questioning or self-contradictory text is refused even though a
    plan reached this point, because the request text is what the actuator
    ultimately executes.
    """

    def __init__(self, telemetry: SupportsSnapshotWrite) -> None:
        self._telemetry = telemetry

    async def apply(self, request: str, *, machine: MachineSnapshot) -> ControlOutcome:
        command = control_command(request)
        if command is ControlIntent.QUESTION:
            return self._refused(request, "该请求是询问语气，未被当作操作指令。")
        if command is ControlIntent.CONFLICT:
            return self._refused(request, "指令同时包含启动与停止，请分开下达。")
        if command is ControlIntent.STOP:
            updated = dataclasses.replace(
                machine, engine_running=False, operating_state="熄火"
            )
            self._commit(updated)
            return ControlOutcome(
                accepted=True,
                summary=(
                    f"已通过虚拟安全网关执行：{request}。"
                    "发动机已停止，运行状态更新为「熄火」。"
                ),
                machine=updated,
            )
        if command is ControlIntent.START:
            updated = dataclasses.replace(
                machine, engine_running=True, operating_state="怠速"
            )
            self._commit(updated)
            return ControlOutcome(
                accepted=True,
                summary=(
                    f"已通过虚拟安全网关执行：{request}。"
                    "发动机已启动，运行状态更新为「怠速」。"
                ),
                machine=updated,
            )
        if has_negation_cue(request):
            return self._refused(request, "检测到否定或取消语义，未执行设备动作。")
        # A control request the simulator cannot map to a snapshot field is still
        # accepted and recorded, so the full chain stays connected end to end.
        return ControlOutcome(
            accepted=True,
            summary=(
                f"已通过虚拟安全网关下发指令：{request}。"
                "（仿真执行器已记录该动作，未改变具体遥测字段。）"
            ),
            machine=None,
        )

    @staticmethod
    def _refused(request: str, reason: str) -> ControlOutcome:
        return ControlOutcome(
            accepted=False,
            summary=f"虚拟安全网关拒绝执行：{request}。{reason}",
            machine=None,
        )

    def _commit(self, snapshot: MachineSnapshot) -> None:
        self._telemetry.apply_snapshot(snapshot)
