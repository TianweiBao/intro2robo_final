"""Domain randomization for training (Phase 3+) and sim-to-real robustness.

randomize(sim, rng) resets the simulation and then perturbs:
- which tube holds which color, and the target color
- tube volumes
- rack and dish poses (no overlaps)
- material colors, lighting and liquid colors (hues stay recognizable)
- poses and field of view of the wrist and side cameras

sim.reset() restores every nominal value.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import specs as S
from .sim import COLORS, LABWARE_BODIES, TUBES, PipetteSim

FOOTPRINT_HALF = {  # (half x, half y) of each labware footprint on the bench
    "tube_rack": S.TUBE_RACK_HALF,
    "dish": (S.DISH.radius + S.DISH_WALL, S.DISH.radius + S.DISH_WALL),
}
RANDOMIZED_MATERIALS = ("bench", "floor", "rack_white", "accent_purple", "kinova_white", "pipette_white",
                        "aluminum", "actuator_gray")


@dataclass
class RandomizationConfig:
    shuffle_colors: bool = True          # tube positions of red / blue / green
    labware_xy: float = 0.02             # +- m per labware item
    labware_yaw_deg: float = 10.0        # +- deg
    tube_volume_ul: tuple = (20000.0, 45000.0)
    color_jitter: float = 0.08           # +- on material RGB
    liquid_color_jitter: float = 0.04    # small, so red / blue / green stay distinguishable
    light_scale: tuple = (0.6, 1.3)
    light_pos_jitter: float = 0.3
    side_cam_pos: float = 0.02           # m
    side_cam_rot_deg: float = 2.0
    wrist_cam_pos: float = 0.002         # mounting error of the D405 bracket
    wrist_cam_rot_deg: float = 1.0
    fovy_jitter_deg: float = 1.5


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _rects_overlap(c1, h1, yaw1, c2, h2, yaw2, margin: float) -> bool:
    """Separating-axis test for two oriented rectangles (centers, half extents, yaw)."""
    axes = [np.array([np.cos(a), np.sin(a)]) for a in (yaw1, yaw1 + np.pi / 2, yaw2, yaw2 + np.pi / 2)]
    d = np.asarray(c2) - np.asarray(c1)

    def radius(h, yaw, ax):
        u, v = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        return h[0] * abs(u @ ax) + h[1] * abs(v @ ax)

    return all(abs(d @ ax) < radius(h1, yaw1, ax) + radius(h2, yaw2, ax) + margin for ax in axes)


def _small_rotation(rng, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, np.radians(rng.uniform(-max_deg, max_deg)))
    return q


def randomize(sim: PipetteSim, rng: np.random.Generator, cfg: RandomizationConfig | None = None,
              target_color: str | None = None) -> dict:
    """Reset `sim` with randomized scene parameters. Returns the sampled values."""
    cfg = cfg or RandomizationConfig()
    m = sim.model
    colors = tuple(str(c) for c in rng.permutation(COLORS)) if cfg.shuffle_colors else tuple(S.DEFAULT_TUBE_COLORS)
    info = {
        "tube_colors": colors,
        "target_color": target_color or str(rng.choice(COLORS)),
        "tube_volumes": rng.uniform(*cfg.tube_volume_ul, len(TUBES)),
    }
    sim.reset(info["tube_volumes"], colors, 0.0, info["target_color"])

    # labware poses, resampled until the footprints don't overlap (1 cm margin)
    nominal = {n: (m.body_pos[m.body(n).id].copy(), m.body_quat[m.body(n).id].copy()) for n in LABWARE_BODIES}
    for _ in range(100):
        poses = {}
        for n in LABWARE_BODIES:
            pos, quat = nominal[n]
            yaw = 2 * np.arctan2(quat[3], quat[0]) + np.radians(rng.uniform(-cfg.labware_yaw_deg, cfg.labware_yaw_deg))
            poses[n] = (pos[:2] + rng.uniform(-cfg.labware_xy, cfg.labware_xy, 2), yaw)
        (a, pa), (b, pb) = poses.items()
        if not _rects_overlap(pa[0], FOOTPRINT_HALF[a], pa[1], pb[0], FOOTPRINT_HALF[b], pb[1], 0.01):
            break
    for n, (xy, yaw) in poses.items():
        b = m.body(n).id
        m.body_pos[b, :2] = xy
        m.body_quat[b] = _yaw_quat(yaw)
    info["labware"] = {n: (xy.tolist(), float(np.degrees(yaw))) for n, (xy, yaw) in poses.items()}

    # appearance
    for name in RANDOMIZED_MATERIALS:
        mid = m.material(name).id
        m.mat_rgba[mid, :3] = np.clip(m.mat_rgba[mid, :3] + rng.uniform(-cfg.color_jitter, cfg.color_jitter, 3), 0, 1)
    for c in sim.containers.values():
        c.rgb = np.clip(c.rgb + rng.uniform(-cfg.liquid_color_jitter, cfg.liquid_color_jitter, 3), 0, 1)
        sim.visuals.update_container(c)
    for lid in range(m.nlight):
        m.light_diffuse[lid] *= rng.uniform(*cfg.light_scale)
        m.light_pos[lid] += rng.uniform(-cfg.light_pos_jitter, cfg.light_pos_jitter, 3)

    # cameras: side camera placement and wrist-camera mounting errors
    for cam, dpos, drot in (("side", cfg.side_cam_pos, cfg.side_cam_rot_deg),
                            ("wrist", cfg.wrist_cam_pos, cfg.wrist_cam_rot_deg)):
        cid = m.cam(cam).id
        m.cam_pos[cid] += rng.uniform(-dpos, dpos, 3)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, m.cam_quat[cid].copy(), _small_rotation(rng, drot))
        m.cam_quat[cid] = q
        m.cam_fovy[cid] += rng.uniform(-cfg.fovy_jitter_deg, cfg.fovy_jitter_deg)

    mujoco.mj_forward(m, sim.data)
    return info
