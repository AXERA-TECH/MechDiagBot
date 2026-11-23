from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cat_assistant.adapters.simulation import (
    ScenarioTelemetry,
    get_simulation_scenario,
    simulation_catalog,
)
from cat_assistant.bootstrap import build_demo_app
from cat_assistant.domain.models import ResponseCategory, Utterance


class SimulationScenarioTests(unittest.IsolatedAsyncioTestCase):
    def test_catalog_is_replayable_and_has_multi_stage_signals(self) -> None:
        catalog = simulation_catalog()
        self.assertGreaterEqual(len(catalog), 3)
        for scenario in catalog.values():
            self.assertGreaterEqual(len(scenario.samples), 4)
            self.assertEqual(
                tuple(sample.observed_at for sample in scenario.samples),
                tuple(sorted(sample.observed_at for sample in scenario.samples)),
            )
            self.assertTrue(any(sample.fault_codes for sample in scenario.samples) or scenario.scenario_id == "fuel-and-service-due")

    async def test_scenario_telemetry_replays_and_clamps_phase(self) -> None:
        scenario = get_simulation_scenario("hydraulic-pressure-drift")
        telemetry = ScenarioTelemetry(scenario)
        first = await telemetry.get_snapshot("cat-306-demo")
        self.assertEqual(first.operating_state, "作业中")
        telemetry.set_phase(2)
        alarm = await telemetry.get_snapshot("cat-306-demo")
        self.assertEqual(alarm.fault_codes, ("E123",))
        self.assertEqual(alarm.fuel_percent, 66)
        telemetry.set_phase(99)
        final = await telemetry.get_snapshot("cat-306-demo")
        self.assertEqual(final.operating_state, "待检修")

    async def test_complex_scenario_drives_existing_diagnostic_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_demo_app(
                Path(directory) / "events.jsonl",
                simulation_scenario="hydraulic-pressure-drift",
                simulation_phase=2,
            )
            response = await app.handle(
                Utterance(
                    "查看当前状态，查询 E123 维修手册，给出处理方案",
                    "simulation-session",
                    "cat-306-demo",
                    "operator-1",
                )
            )
            self.assertEqual(response.category, ResponseCategory.DIAGNOSTIC)
            self.assertIn("E123", response.text)
            self.assertIn("Service Manual", response.text)
            self.assertEqual(response.metadata["steps"], ["observe", "state", "retrieve", "diagnose", "propose"])

    async def test_unknown_scenario_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_demo_app(simulation_scenario="not-a-real-incident")


if __name__ == "__main__":
    unittest.main()
