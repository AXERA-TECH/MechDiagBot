from __future__ import annotations

import dataclasses
import unittest

from cat_assistant.adapters.telemetry import (
    DemoFleetTelemetry,
    _FLEET_FAULT_POOL,
    demo_snapshots,
    fault_code_catalog,
)
from cat_assistant.domain.models import MachineSnapshot


def _dot_tone(machine: MachineSnapshot) -> str:
    """Mirror the frontend indicator logic (app.js ``dotTone``)."""
    if not machine.engine_running:
        return "off"
    return "fault" if machine.fault_codes else "ok"


def _tones(snapshots: tuple[MachineSnapshot, ...]) -> dict[str, str]:
    return {snapshot.machine_id: _dot_tone(snapshot) for snapshot in snapshots}


def _stop(telemetry: DemoFleetTelemetry, machine_id: str) -> None:
    """Simulate a 停止 control action pinning the machine to a shut-down state."""
    base = telemetry._snapshots[machine_id]
    telemetry.apply_snapshot(
        dataclasses.replace(base, engine_running=False, operating_state="熄火")
    )


class DemoFleetOverlayTests(unittest.IsolatedAsyncioTestCase):
    def test_unpinned_profile_is_two_fault_one_ok_one_off_across_windows(self) -> None:
        telemetry = DemoFleetTelemetry(demo_snapshots())
        for window in range(2000):
            telemetry._window = lambda w=window: w
            counts: dict[str, int] = {"fault": 0, "ok": 0, "off": 0}
            for tone in _tones(telemetry.list_snapshots()).values():
                counts[tone] += 1
            self.assertEqual(
                counts,
                {"fault": 2, "ok": 1, "off": 1},
                f"window {window} broke the 2/1/1 profile: {counts}",
            )

    async def test_single_snapshot_agrees_with_fleet_list(self) -> None:
        telemetry = DemoFleetTelemetry(demo_snapshots())
        for window in range(500):
            telemetry._window = lambda w=window: w
            listed = _tones(telemetry.list_snapshots())
            for machine_id, tone in listed.items():
                single = await telemetry.get_snapshot(machine_id)
                self.assertEqual(
                    _dot_tone(single),
                    tone,
                    f"window {window}: {machine_id} disagreed (list={tone}, "
                    f"single={_dot_tone(single)})",
                )

    def test_at_least_one_fault_holds_after_stopping_faulted_machines(self) -> None:
        # Across every window: stopping *both* overlay-faulted machines (pinning
        # them gray) must still leave >=1 machine showing a fault — the floor.
        for window in range(2000):
            telemetry = DemoFleetTelemetry(demo_snapshots())
            telemetry._window = lambda w=window: w
            faulted = [
                machine_id
                for machine_id, tone in _tones(telemetry.list_snapshots()).items()
                if tone == "fault"
            ]
            self.assertEqual(len(faulted), 2, f"window {window}: expected 2 faults")
            for machine_id in faulted:
                _stop(telemetry, machine_id)
            after = _tones(telemetry.list_snapshots())
            faults_after = [m for m, tone in after.items() if tone == "fault"]
            self.assertGreaterEqual(
                len(faults_after),
                1,
                f"window {window}: floor breached, tones={after}",
            )
            # The two machines the operator stopped stay stopped (real control
            # state is honoured, never overridden by the floor).
            for machine_id in faulted:
                self.assertEqual(after[machine_id], "off")

    async def test_actuated_machine_is_reported_verbatim(self) -> None:
        telemetry = DemoFleetTelemetry(demo_snapshots())
        telemetry._window = lambda: 3
        target = "cat-306-demo"
        pinned = dataclasses.replace(
            telemetry._snapshots[target],
            engine_running=False,
            operating_state="熄火",
            fault_codes=(),
        )
        telemetry.apply_snapshot(pinned)

        single = await telemetry.get_snapshot(target)
        self.assertFalse(single.engine_running)
        self.assertEqual(single.operating_state, "熄火")
        listed = {s.machine_id: s for s in telemetry.list_snapshots()}[target]
        self.assertFalse(listed.engine_running)
        self.assertEqual(listed.operating_state, "熄火")

    def test_floor_cannot_override_a_fully_pinned_healthy_fleet(self) -> None:
        # If the operator has taken over *every* machine to a non-fault state,
        # the overlay steps aside entirely rather than fabricating a fault.
        telemetry = DemoFleetTelemetry(demo_snapshots())
        telemetry._window = lambda: 11
        for machine_id in list(telemetry._snapshots):
            base = telemetry._snapshots[machine_id]
            telemetry.apply_snapshot(
                dataclasses.replace(
                    base, engine_running=True, operating_state="怠速", fault_codes=()
                )
            )
        tones = set(_tones(telemetry.list_snapshots()).values())
        self.assertEqual(tones, {"ok"})


class FaultCodeCatalogTests(unittest.TestCase):
    def test_catalog_covers_every_code_that_can_surface(self) -> None:
        catalog = fault_code_catalog()
        pool_codes = {code for codes in _FLEET_FAULT_POOL.values() for code in codes}
        snapshot_codes = {
            code for snap in demo_snapshots().values() for code in snap.fault_codes
        }
        missing = (pool_codes | snapshot_codes) - set(catalog)
        self.assertEqual(missing, set(), f"catalog missing descriptions for {missing}")

    def test_every_description_is_non_empty(self) -> None:
        for code, description in fault_code_catalog().items():
            self.assertTrue(description.strip(), f"empty description for {code}")

    def test_accessor_returns_an_isolated_copy(self) -> None:
        first = fault_code_catalog()
        first["E123"] = "mutated"
        self.assertNotEqual(fault_code_catalog().get("E123"), "mutated")


if __name__ == "__main__":
    unittest.main()
