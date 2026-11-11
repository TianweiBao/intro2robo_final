"""Geometry and constants shared by the scene builder and the runtime simulation.

Units: meters, kilograms, seconds; liquid volumes in microliters (uL).
The pipette numbers must match models/gen3_lite_pipette.xml (pipette frame: +z runs down the pipette).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

UL = 1e-9  # cubic meters per microliter

# ----------------------------------------------------------------------------------------------
# Pipette: standard manual 1000 uL pipette, plunger pressed by a linear actuator, tip always on
# ----------------------------------------------------------------------------------------------
NOZZLE_Z = 0.203                   # nozzle end, measured down the pipette from the flange level
TIP_SEAT_DEPTH = 0.012             # the nozzle end sits this far inside the tip
TIP_LENGTH = 0.076
TIP_TOP_Z = NOZZLE_Z - TIP_SEAT_DEPTH          # 0.191
TIP_END_Z = TIP_TOP_Z + TIP_LENGTH             # 0.267
PIPETTE_AXIS_OFFSET = 0.075        # pipette axis in front of the flange face center

PLUNGER_FIRST_STOP = 0.010         # plunger travel (m) that displaces the nominal 1000 uL
PLUNGER_BLOWOUT = 0.013            # second stop
PLUNGER_MAX_SPEED = 0.010          # m/s, linear actuator speed (~1 mL/s)
PIPETTE_VOLUME_UL = 1000.0
UL_PER_M_PLUNGER = PIPETTE_VOLUME_UL / PLUNGER_FIRST_STOP
MIN_IMMERSION = 0.0005             # tip end must be at least this far below the surface to draw liquid

# ----------------------------------------------------------------------------------------------
# 1000 uL tip. Tip body frame: origin at the top (opening) center, +z toward the tip end.
# ----------------------------------------------------------------------------------------------
TIP_COLLAR_HEIGHT = 0.010
# Inner bore as (height above the tip end, inner radius). Holds ~1.3 mL below the nozzle, so
# 1000 uL fills ~55 mm of the tip and leaves an air cushion, like a real tip.
TIP_BORE_PROFILE = ((0.0, 0.00025), (0.035, 0.0030), (TIP_LENGTH - TIP_COLLAR_HEIGHT, 0.0035))
# Liquid in the tip is drawn as a stack of cylinders that fit inside the bore: (z0, z1, radius).
TIP_LIQUID_SEGMENTS = ((0.0, 0.012, 0.0006), (0.012, 0.024, 0.0015), (0.024, 0.035, 0.0024),
                       (0.035, TIP_LENGTH - TIP_SEAT_DEPTH, 0.0029))


def _bore_volume(h: float) -> float:
    """Volume (uL) of the tip bore from the tip end up to height h."""
    v = 0.0
    for (z0, r0), (z1, r1) in zip(TIP_BORE_PROFILE[:-1], TIP_BORE_PROFILE[1:]):
        if h <= z0:
            break
        top = min(h, z1)
        rt = r0 + (r1 - r0) * (top - z0) / (z1 - z0)
        v += math.pi * (top - z0) * (r0 * r0 + r0 * rt + rt * rt) / 3.0
    return v / UL


def tip_liquid_height(volume_ul: float) -> float:
    """Height of a liquid column of `volume_ul`, measured from the tip end."""
    if volume_ul <= 0:
        return 0.0
    lo, hi = 0.0, TIP_BORE_PROFILE[-1][0]
    for _ in range(40):
        h = 0.5 * (lo + hi)
        lo, hi = (h, hi) if _bore_volume(h) < volume_ul else (lo, h)
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ContainerSpec:
    """Inner geometry of a cylindrical liquid container, measured from its lowest inner point (the "floor")."""

    radius: float                    # inner radius
    depth: float                     # floor to rim
    cone_height: float = 0.0         # conical bottom (tubes); counted inside `depth`

    @property
    def area(self) -> float:
        return math.pi * self.radius**2

    def volume_at_height(self, h: float) -> float:
        """Liquid volume (uL) when the surface is `h` above the floor."""
        h = min(max(h, 0.0), self.depth)
        if self.cone_height > 0.0:
            hc = self.cone_height
            if h <= hc:
                return self.area * hc / 3.0 * (h / hc) ** 3 / UL
            return (self.area * hc / 3.0 + self.area * (h - hc)) / UL
        return self.area * h / UL

    def height_of_volume(self, volume_ul: float) -> float:
        """Surface height above the floor for `volume_ul` of liquid."""
        v = max(volume_ul, 0.0) * UL
        if self.cone_height > 0.0:
            vc = self.area * self.cone_height / 3.0
            if v <= vc:
                return self.cone_height * (v / vc) ** (1.0 / 3.0)
            return self.cone_height + (v - vc) / self.area
        return v / self.area

    @property
    def capacity_ul(self) -> float:
        return self.volume_at_height(self.depth)

    def inside_xy(self, dx: float, dy: float, margin: float = 0.0) -> bool:
        """Is the point (dx, dy), in the container frame, inside the opening (shrunk by `margin`)?"""
        return math.hypot(dx, dy) <= self.radius - margin


# Source: one row of three 50 mL conical tubes (e.g. Falcon 352070: ~29.1 mm OD, 115 mm long).
TUBE_50ML = ContainerSpec(radius=0.0135, depth=0.114, cone_height=0.020)
TUBE_WALL = 0.0012
# Visual-only colored annulus at each tube rim. The ring is the intended camera target for
# identifying and centering over a tube without obscuring its open interior.
TUBE_TARGET_RING_WIDTH = 0.0035
TUBE_TARGET_RING_HEIGHT = 0.0030
N_TUBES = 3
TUBE_RACK_PITCH = 0.042
TUBE_RACK_BASE = 0.004             # tubes stand on this plate
TUBE_FLOOR_Z = TUBE_RACK_BASE + 0.001
TUBE_RACK_HALF = (0.066, 0.024)    # footprint half extents (along the row, across it)

# Destination: clear glass dish, 70 mm across and 45 mm tall.
DISH = ContainerSpec(radius=0.033, depth=0.042)
DISH_WALL = 0.002
DISH_FLOOR_Z = 0.003               # glass bottom thickness
DISH_HEIGHT = DISH_FLOOR_Z + DISH.depth

# Liquids are identified by color (food-dye-like colored water).
LIQUID_COLORS = {"red": (0.82, 0.10, 0.12), "blue": (0.12, 0.30, 0.86), "green": (0.10, 0.62, 0.22)}
DEFAULT_TUBE_COLORS = ("red", "blue", "green")   # tube_1 (left) .. tube_3 (right), seen from the front
LIQUID_ALPHA = 0.85


def tube_xy(index: int) -> tuple[float, float]:
    """Tube center in the rack frame; tube 0 is at -x (left as seen from the front of the bench)."""
    return ((index - (N_TUBES - 1) / 2.0) * TUBE_RACK_PITCH, 0.0)
