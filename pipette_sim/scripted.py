"""Scripted three-phase transfer, as in the task picture:

    1 Aspirate   go to the tube holding the target color, draw the liquid
    2 Transfer   carry it along an arc to above the dish
    3 Dispense   lower into the dish and blow it out, then return to the ready pose

The script reads the source tube from the simulator (privileged). A vision policy has to find the
tube by its color in the camera images. Phase 2 turns this into the scripted expert that records
demonstrations.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import specs as S
from .motion import MotionPrimitives

PHASES = ("1 Aspirate", "2 Transfer", "3 Dispense")
READY_TCP = np.array([0.30, 0.0, 0.25])  # tip-end position of the ready pose
TRAVEL_Z = 0.17          # tip-end height while moving between stations (tube rims at 0.119)
APPROACH = 0.03          # start of the slow vertical descent, above the rim
ARC_LIFT = 0.05          # how far the transfer arc rises above TRAVEL_Z
FAST, SLOW = 0.15, 0.025  # m/s
IMMERSION = 0.003        # tip end this far below the surface while aspirating
DISPENSE_HEIGHT = 0.005  # tip end this far above the dish floor (or its liquid) while dispensing


def _above(mp: MotionPrimitives, name: str, z: float) -> np.ndarray:
    floor, _ = mp.sim.container_frame(name)
    return np.array([floor[0], floor[1], z])


def aspirate(mp: MotionPrimitives, tube: str, volume_ul: float = S.PIPETTE_VOLUME_UL) -> float:
    sim = mp.sim
    mp.move_to(_above(mp, tube, TRAVEL_Z), speed=FAST)
    mp.move_to(_above(mp, tube, sim.rim_z(tube) + APPROACH), speed=FAST)
    mp.plunger_to(volume_ul / S.UL_PER_M_PLUNGER)          # press to the volume stop while still in air
    floor_z = sim.container_frame(tube)[0][2]
    depth = max(sim.surface_z(tube) - IMMERSION, floor_z + S.TUBE_50ML.cone_height + 0.002)
    mp.move_to(_above(mp, tube, depth), speed=SLOW)
    before = sim.tip_volume
    mp.plunger_to(0.0, speed=0.003)                        # slow release draws the liquid in
    mp.wait(0.5)
    mp.move_to(_above(mp, tube, TRAVEL_Z), speed=0.08)
    return sim.tip_volume - before


def transfer(mp: MotionPrimitives, dest: str = "dish") -> None:
    mp.move_arc(_above(mp, dest, TRAVEL_Z), lift=ARC_LIFT, speed=0.12)


def dispense(mp: MotionPrimitives, dest: str = "dish") -> float:
    sim = mp.sim
    floor_z = sim.container_frame(dest)[0][2]
    level = min(max(floor_z + DISPENSE_HEIGHT, sim.surface_z(dest) + 0.003), sim.rim_z(dest) - 0.005)
    mp.move_to(_above(mp, dest, sim.rim_z(dest) + APPROACH), speed=FAST)
    mp.move_to(_above(mp, dest, level), speed=SLOW * 2)
    before = sim.tip_volume
    mp.plunger_to(S.PLUNGER_BLOWOUT, speed=0.004)          # through the first stop to blow-out
    mp.move_to(_above(mp, dest, TRAVEL_Z), speed=0.10)      # withdraw with the plunger held down
    mp.plunger_to(0.0, speed=0.008)
    return before - sim.tip_volume


def run_task(mp: MotionPrimitives, color: str | None = None, dest: str = "dish",
             volume_ul: float = S.PIPETTE_VOLUME_UL, on_phase: Callable[[str], None] | None = None) -> dict:
    """Move `volume_ul` of the `color` liquid (default: the simulator's target color) into `dest`."""
    sim = mp.sim
    on_phase = on_phase or (lambda name: None)
    if color is not None:
        sim.target_color = color
    source = sim.source_tube
    v_src, v_dst = sim.containers[source].volume, sim.containers[dest].volume
    on_phase(PHASES[0])
    drawn = aspirate(mp, source, volume_ul)
    on_phase(PHASES[1])
    transfer(mp, dest)
    on_phase(PHASES[2])
    delivered = dispense(mp, dest)
    on_phase("return")
    mp.move_to(READY_TCP, speed=FAST)
    return {"color": sim.target_color, "source": source, "aspirated_ul": round(drawn, 1),
            "dispensed_ul": round(delivered, 1),
            "source_change_ul": round(sim.containers[source].volume - v_src, 1),
            "dest_change_ul": round(sim.containers[dest].volume - v_dst, 1)}
