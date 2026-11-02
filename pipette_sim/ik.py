"""Inverse kinematics for the Gen3 lite with the pipette held vertically.

The target is the position of a tool site ("tip_end" or "nozzle") plus a heading angle. Both sites
carry the pipette frame: z runs down the pipette, x points from the flange toward the pipette (and
on to the wrist camera). IK keeps site z pointing straight down and site x at the requested world
heading.

Default heading: tangential, 90 deg clockwise from the radial direction. The flange then sits
beside the pipette, on its right as seen from the front of the bench. With the flange pointing
radially outward instead, the wrist folds back against the robot and joints end up near their
limits.
"""
from __future__ import annotations

import numpy as np
import mujoco

ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 7))
HEADING_OFFSET = -np.pi / 2

# Elbow-up posture, pipette vertical, tip end at (0.30, 0, 0.25), default heading. Default IK seed.
READY_Q = np.array([0.5988, -0.3574, 0.9429, -0.1809, 2.1462, -0.3222])


def default_heading(target: np.ndarray) -> float:
    """Heading of the pipette-frame x axis for a target position (tangential, clockwise)."""
    return float(np.arctan2(target[1], target[0]) + HEADING_OFFSET)


class VerticalIK:
    """Damped-least-squares IK on a private MjData, so it never disturbs the simulation state."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        jids = [model.joint(j).id for j in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()

    def fk(self, q: np.ndarray, site: str = "tip_end") -> tuple[np.ndarray, np.ndarray]:
        """Site position and site (pipette-frame) rotation matrix for arm joints q."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site(site).xpos.copy(), self.d.site(site).xmat.reshape(3, 3).copy()

    def solve(self, target: np.ndarray, heading: float | None = None, site: str = "tip_end",
              q_init: np.ndarray | None = None, iters: int = 300, tol_pos: float = 2e-4,
              tol_rot: float = 2e-3) -> tuple[np.ndarray, bool]:
        """Single DLS solve from q_init. Returns (q, converged)."""
        target = np.asarray(target, float)
        if heading is None:
            heading = default_heading(target)
        z_des = np.array([0.0, 0.0, -1.0])
        x_des = np.array([np.cos(heading), np.sin(heading), 0.0])
        sid = self.m.site(site).id
        q = np.clip(np.array(READY_Q if q_init is None else q_init, float), self.lo, self.hi)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            R = self.d.site_xmat[sid].reshape(3, 3)
            e_pos = target - self.d.site_xpos[sid]
            e_rot = 0.5 * (np.cross(R[:, 2], z_des) + np.cross(R[:, 0], x_des))
            if np.linalg.norm(e_pos) < tol_pos and np.linalg.norm(e_rot) < tol_rot:
                return q, True
            mujoco.mj_jacSite(self.m, self.d, jp, jr, sid)
            J = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
            err = np.concatenate([e_pos, e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            step = np.max(np.abs(dq))
            if step > 0.3:  # limit the step so the solver follows the local branch
                dq *= 0.3 / step
            q = np.clip(q + dq, self.lo, self.hi)
        return q, False

    def solve_robust(self, target: np.ndarray, heading: float | None = None, site: str = "tip_end",
                     q_init: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """Try q_init, then the ready posture turned toward the target, then perturbed seeds."""
        target = np.asarray(target, float)
        turned = READY_Q.copy()
        turned[0] += float(np.arctan2(target[1], target[0]))
        seeds = [s for s in (q_init, turned, READY_Q) if s is not None]
        rng = np.random.default_rng(0)
        seeds += [turned + rng.normal(0, 0.3, 6) for _ in range(8)]
        best = None
        q = seeds[0]
        for s in seeds:
            q, ok = self.solve(target, heading, site, s)
            if ok:
                margin = np.min(np.minimum(q - self.lo, self.hi - q))
                if best is None or margin > best[1]:
                    best = (q, margin)
                if margin > 0.3:
                    break
        if best is None:
            return q, False
        return best[0], True
