"""Scripted motion primitives: straight-line and arc moves of the pipette tip end with the pipette
kept vertical, plus plunger moves.

Each control step solves IK for the next waypoint (warm-started from the previous solution) and
sends the joint angles to the arm's position actuators. Used by the scripted task, the checks and
the viewer demo.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .ik import VerticalIK, default_heading
from .sim import PipetteSim


def _min_jerk(s: float) -> float:
    return s * s * s * (10 - 15 * s + 6 * s * s)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class MotionPrimitives:
    def __init__(self, sim: PipetteSim, on_step: Callable[[], None] | None = None):
        self.sim = sim
        self.ik = VerticalIK(sim.model)
        self.on_step = on_step or (lambda: None)
        self.q_cmd = sim.arm_q

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.sim.step()
            self.on_step()

    def _heading_now(self) -> float:
        x = self.sim.tool_frame()[:, 0]
        return math.atan2(x[1], x[0])

    def _follow(self, path: Callable[[float], np.ndarray], length: float, target: np.ndarray, speed: float,
                heading: float | None, settle_tol: float, settle_time: float) -> float:
        """Track path(s), s in [0, 1], with min-jerk timing; returns the final tip-end error (m)."""
        h0 = self._heading_now()
        dh = _wrap((default_heading(target) if heading is None else heading) - h0)
        duration = max(length / speed * 1.875, abs(dh) / 0.8, self.sim.control_dt)  # 1.875 = min-jerk peak/avg
        n = max(1, math.ceil(duration / self.sim.control_dt))
        for k in range(1, n + 1):
            s = _min_jerk(k / n)
            q, ok = self.ik.solve(path(s), h0 + s * dh, "tip_end", self.q_cmd, iters=60)
            if not ok:
                q, ok = self.ik.solve_robust(path(s), h0 + s * dh, "tip_end", self.q_cmd)
            if ok:
                self.q_cmd = q
            self.sim.set_arm_targets(self.q_cmd)
            self.step()
        for _ in range(int(settle_time / self.sim.control_dt)):
            if np.linalg.norm(self.sim.tcp() - target) < settle_tol:
                break
            self.step()
        return float(np.linalg.norm(self.sim.tcp() - target))

    def move_to(self, target, speed: float = 0.10, heading: float | None = None, settle_tol: float = 5e-4,
                settle_time: float = 1.0) -> float:
        """Straight line from the commanded tip-end position to `target`."""
        target = np.asarray(target, float)
        start, _ = self.ik.fk(self.q_cmd)
        return self._follow(lambda s: start + s * (target - start), float(np.linalg.norm(target - start)),
                            target, speed, heading, settle_tol, settle_time)

    def move_arc(self, target, lift: float = 0.05, speed: float = 0.12, heading: float | None = None,
                 settle_tol: float = 5e-4, settle_time: float = 1.0) -> float:
        """Arc from the commanded tip-end position to `target`, rising `lift` above the straight line midway."""
        target = np.asarray(target, float)
        start, _ = self.ik.fk(self.q_cmd)
        up = np.array([0.0, 0.0, 1.0])
        path = lambda s: start + s * (target - start) + 4.0 * lift * s * (1.0 - s) * up
        length = float(np.linalg.norm(target - start)) + 2.0 * lift
        return self._follow(path, length, target, speed, heading, settle_tol, settle_time)

    def wait(self, seconds: float) -> None:
        self.step(max(1, round(seconds / self.sim.control_dt)))

    def plunger_to(self, travel: float, speed: float = 0.004) -> None:
        """Ramp the plunger to `travel` (m) at `speed` (m/s); 4 mm/s ~ 400 uL/s."""
        x0 = self.sim.plunger_target
        n = max(1, math.ceil(abs(travel - x0) / (speed * self.sim.control_dt)))
        for k in range(1, n + 1):
            self.sim.set_plunger(x0 + (travel - x0) * k / n)
            self.step()
        self.wait(0.25)
