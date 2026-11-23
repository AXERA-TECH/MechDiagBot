from __future__ import annotations

import dataclasses
import unittest

from cat_assistant.adapters.capabilities import MachineControlProvider
from cat_assistant.adapters.control import SimulatedMachineControl
from cat_assistant.adapters.telemetry import InMemoryTelemetry, demo_snapshot
from cat_assistant.application.capabilities import CapabilityExecutionContext
from cat_assistant.domain.models import Utterance
from cat_assistant.domain.plans import TaskStep


def _context(machine):
    return CapabilityExecutionContext(
        Utterance("停止运行机器", "s", machine.machine_id, "op"), machine
    )


def _step(request: str) -> TaskStep:
    return TaskStep("act", "machine.control", {"request": request})


class SimulatedMachineControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_command_updates_snapshot_and_reports_new_state(self) -> None:
        machine = demo_snapshot()
        telemetry = InMemoryTelemetry({machine.machine_id: machine})
        executor = SimulatedMachineControl(telemetry)

        outcome = await executor.apply("停止运行机器", machine=machine)

        self.assertTrue(outcome.accepted)
        self.assertIsNotNone(outcome.machine)
        self.assertFalse(outcome.machine.engine_running)
        self.assertEqual(outcome.machine.operating_state, "熄火")
        # The write is authoritative for the next telemetry read.
        after = await telemetry.get_snapshot(machine.machine_id)
        self.assertFalse(after.engine_running)
        self.assertEqual(after.operating_state, "熄火")

    async def test_start_command_restarts_engine(self) -> None:
        machine = demo_snapshot()
        stopped = dataclasses.replace(
            machine, engine_running=False, operating_state="熄火"
        )
        telemetry = InMemoryTelemetry({machine.machine_id: stopped})
        executor = SimulatedMachineControl(telemetry)

        outcome = await executor.apply("启动发动机", machine=stopped)

        self.assertTrue(outcome.machine.engine_running)
        self.assertEqual(outcome.machine.operating_state, "怠速")
        after = await telemetry.get_snapshot(machine.machine_id)
        self.assertTrue(after.engine_running)

    async def test_unrecognized_command_is_accepted_without_field_change(self) -> None:
        machine = demo_snapshot()
        telemetry = InMemoryTelemetry({machine.machine_id: machine})
        executor = SimulatedMachineControl(telemetry)

        outcome = await executor.apply("把大臂抬高 10 厘米", machine=machine)

        self.assertTrue(outcome.accepted)
        self.assertIsNone(outcome.machine)
        after = await telemetry.get_snapshot(machine.machine_id)
        self.assertEqual(after.operating_state, machine.operating_state)


class MachineControlProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_without_executor_denies_as_safe_default(self) -> None:
        provider = MachineControlProvider()

        result = await provider.execute(
            {"request": "停止发动机"}, _context(demo_snapshot()), _step("停止发动机")
        )

        self.assertEqual(result.status, "denied")
        self.assertIn("控制执行器", result.content)

    async def test_with_executor_completes_and_emits_action_result(self) -> None:
        machine = demo_snapshot()
        telemetry = InMemoryTelemetry({machine.machine_id: machine})
        provider = MachineControlProvider(SimulatedMachineControl(telemetry))

        result = await provider.execute(
            {"request": "停止发动机"}, _context(machine), _step("停止发动机")
        )

        self.assertEqual(result.status, "completed")
        self.assertIn("虚拟安全网关", result.content)
        self.assertEqual(result.artifacts[0].artifact_type, "action_result")
        self.assertEqual(result.artifacts[0].data["operating_state"], "熄火")
        self.assertFalse(result.artifacts[0].data["engine_running"])


if __name__ == "__main__":
    unittest.main()
