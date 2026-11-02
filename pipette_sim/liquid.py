"""Liquid bookkeeping. MuJoCo does not simulate fluids, so every container (and the pipette tip)
tracks a volume and a color, and its liquid is drawn as non-colliding geoms resized to match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import specs as S

LIQUID_LIFT = 0.0003  # draw liquid this far above the container floor so it doesn't z-fight with it


def liquid_column(spec: S.ContainerSpec, volume_ul: float) -> tuple[bool, float, float]:
    """How to draw `volume_ul` in a container.

    Returns (cone_visible, half_height, center_above_floor) for the cylindrical liquid geom.
    Conical tubes draw the cone part with a fixed cone mesh once it is full; below that a thin
    cylinder approximates the small amount of liquid in the cone.
    """
    if volume_ul <= 1e-6:
        return False, 0.0, 0.0
    h = spec.height_of_volume(volume_ul)
    if spec.cone_height > 0.0 and h >= spec.cone_height:
        col = h - spec.cone_height
        return True, col / 2.0, spec.cone_height + col / 2.0
    return False, h / 2.0, h / 2.0


def mix_rgb(rgb_a, vol_a: float, rgb_b, vol_b: float) -> np.ndarray:
    total = vol_a + vol_b
    if total <= 0:
        return np.asarray(rgb_b, float)
    return (np.asarray(rgb_a, float) * vol_a + np.asarray(rgb_b, float) * vol_b) / total


@dataclass
class Container:
    name: str
    spec: S.ContainerSpec
    body_id: int          # labware body
    floor_site: int       # container frame: lowest inner point, z up
    liquid_geom: int
    cone_geom: int        # -1 if the container has no conical bottom
    volume: float = 0.0   # uL
    rgb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_radius: float = 0.0  # radius of the liquid cylinder as authored in the XML

    def add(self, volume_ul: float, rgb) -> float:
        """Add liquid; returns the volume that overflowed."""
        room = max(self.spec.capacity_ul - self.volume, 0.0)
        taken = min(volume_ul, room)
        self.rgb = mix_rgb(self.rgb, self.volume, rgb, taken)
        self.volume += taken
        return volume_ul - taken


class LiquidVisuals:
    """Writes container / tip liquid state into geom sizes, positions and colors of a MjModel."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model

    def update_container(self, c: Container) -> None:
        m = self.m
        cone_visible, half, center = liquid_column(c.spec, c.volume)
        g = c.liquid_geom
        if half <= 0.0:
            m.geom_rgba[g, 3] = 0.0
        else:
            m.geom_rgba[g, :3] = c.rgb
            m.geom_rgba[g, 3] = S.LIQUID_ALPHA
            m.geom_size[g, 1] = half
            if c.spec.cone_height > 0.0 and not cone_visible:
                # small volume inside the cone: shrink the radius to roughly follow the cone
                m.geom_size[g, 0] = max(c.base_radius * (2.0 * half) / c.spec.cone_height * 0.75, 0.0005)
            else:
                m.geom_size[g, 0] = c.base_radius
            m.geom_pos[g, 2] = m.site_pos[c.floor_site][2] + center + LIQUID_LIFT
            m.geom_rbound[g] = float(np.hypot(m.geom_size[g, 0], m.geom_size[g, 1]))
        if c.cone_geom >= 0:
            m.geom_rgba[c.cone_geom, :3] = c.rgb
            m.geom_rgba[c.cone_geom, 3] = S.LIQUID_ALPHA if cone_visible else 0.0

    def update_tip(self, segment_geoms: list[int], volume_ul: float, rgb) -> None:
        """Fill the stacked cylinders inside the tip from the tip end upward."""
        h = S.tip_liquid_height(volume_ul)
        for g, (z0, z1, _r) in zip(segment_geoms, S.TIP_LIQUID_SEGMENTS):
            filled = min(max(h - z0, 0.0), z1 - z0)
            if filled <= 1e-5:
                self.m.geom_rgba[g, 3] = 0.0
                continue
            self.m.geom_size[g, 1] = filled / 2.0
            self.m.geom_pos[g, 2] = S.TIP_LENGTH - z0 - filled / 2.0
            self.m.geom_rgba[g, :3] = rgb
            self.m.geom_rgba[g, 3] = S.LIQUID_ALPHA
