"""Generate models/scene.xml: lab bench, a source rack of three colored-liquid tubes, a destination
dish, cameras and lights around the Gen3 lite + pipette defined in models/gen3_lite_pipette.xml.

Regenerate with `python scripts/build_scene.py` after changing SceneConfig.

Names used at runtime:
  containers  tube_1..tube_3 (left to right as seen from the front of the bench) and dish; each has
              a site "<name>_floor" at its lowest inner point and a liquid geom "<name>_liquid"
  bodies      tube_rack, dish
  cameras     wrist (on the robot), side (fixed, for the policy), overview and top (documentation)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import specs as S
from .ik import READY_Q
from .liquid import LIQUID_LIFT, liquid_column

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ENV, ARM, TOOL, TIP = 1, 2, 4, 8


@dataclass
class Placement:
    """Labware pose on the bench in polar coordinates around the robot base. The labware's +x axis
    is tangential (counter-clockwise), so the tube rack runs left to right as seen from the front."""

    r: float
    theta_deg: float

    def pose(self) -> tuple[float, float, float]:
        th = math.radians(self.theta_deg)
        return self.r * math.cos(th), self.r * math.sin(th), th + math.pi / 2


@dataclass
class SceneConfig:
    tube_rack: Placement = field(default_factory=lambda: Placement(0.32, -28.0))   # source, front-left
    dish: Placement = field(default_factory=lambda: Placement(0.32, 28.0))         # destination, front-right
    tube_colors: tuple[str, ...] = S.DEFAULT_TUBE_COLORS
    tube_volume_ul: float = 30000.0
    dish_volume_ul: float = 0.0


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def _n(v: float) -> str:
    s = f"{float(v):.6g}"
    return "0" if s in ("-0", "0") else s


def _f(*vals) -> str:
    return " ".join(_n(v) for v in vals)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def look_at(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    """xyaxes for a MuJoCo camera at `pos` looking at `target` (MuJoCo cameras look along -z)."""
    pos, target, up = (np.asarray(v, float) for v in (pos, target, up))
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return _f(*x, *y)


def _ring(r: float, z: float, n: int = 16) -> list[float]:
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out += [r * math.cos(a), r * math.sin(a), z]
    return out


def _shell_mesh(name: str, r_out: float, r_in: float, height: float, n: int = 32) -> str:
    """Visual-only open-topped cylindrical shell (outer wall, rim, inner wall, bottom ring)."""
    verts = []
    for r, z in ((r_out, 0.0), (r_out, height), (r_in, height), (r_in, 0.0)):
        verts += _ring(r, z, n)
    ob, ot, it, ib = 0, n, 2 * n, 3 * n
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        faces += [ob + k, ob + k1, ot + k1, ob + k, ot + k1, ot + k]      # outer wall
        faces += [ot + k, ot + k1, it + k1, ot + k, it + k1, it + k]      # rim
        faces += [it + k, it + k1, ib + k1, it + k, ib + k1, ib + k]      # inner wall
        faces += [ib + k, ib + k1, ob + k1, ib + k, ob + k1, ob + k]      # bottom ring
    return (f'<mesh name="{name}" inertia="shell" vertex="{_f(*verts)}" '
            f'face="{" ".join(str(i) for i in faces)}"/>')


def _ring_walls(cx: float, cy: float, z0: float, z1: float, r_in: float, t: float, n: int = 12) -> list[str]:
    """Open-topped cylindrical wall made of n boxes (collision only), so the tip can go inside."""
    half_w = (r_in + t) * math.tan(math.pi / n)
    rc = r_in + t / 2
    return [f'<geom class="env" type="box" size="{_f(t / 2, half_w, (z1 - z0) / 2)}" '
            f'pos="{_f(cx + rc * math.cos(a), cy + rc * math.sin(a), (z0 + z1) / 2)}" euler="0 0 {_n(a)}"/>'
            for a in (2 * math.pi * k / n for k in range(n))]


def _liquid(name: str, spec: S.ContainerSpec, volume: float, cx: float, cy: float, floor: float, rgb) -> list[str]:
    """Liquid geoms for a container holding `volume` uL (resized at runtime)."""
    cone_visible, half, center = liquid_column(spec, volume)
    rgba = lambda visible: _f(*rgb, S.LIQUID_ALPHA if visible else 0.0)
    out = []
    if spec.cone_height > 0:
        out.append(f'<geom name="{name}_liquid_cone" class="liquid" type="mesh" mesh="liquid_cone" '
                   f'pos="{_f(cx, cy, floor + spec.cone_height)}" rgba="{rgba(cone_visible)}"/>')
    out.append(f'<geom name="{name}_liquid" class="liquid" size="{_f(spec.radius - 0.0002, max(half, 0.0005))}" '
               f'pos="{_f(cx, cy, floor + max(center, 0.0005) + LIQUID_LIFT)}" rgba="{rgba(half > 0)}"/>')
    return out


# ----------------------------------------------------------------------------------------------
# labware
# ----------------------------------------------------------------------------------------------
def _tube_rack(cfg: SceneConfig) -> list[str]:
    x, y, yaw = cfg.tube_rack.pose()
    spec = S.TUBE_50ML
    hx, hy = S.TUBE_RACK_HALF
    deck_z, floor = 0.062, S.TUBE_FLOOR_Z
    cone_top, rim = floor + spec.cone_height, floor + spec.depth
    out = [f'<body name="tube_rack" pos="{_f(x, y, 0)}" quat="{_f(*_yaw_quat(yaw))}">',
           f'<geom name="rack_base" class="env_both" type="box" size="{_f(hx, hy, S.TUBE_RACK_BASE / 2)}" '
           f'pos="{_f(0, 0, S.TUBE_RACK_BASE / 2)}" material="rack_white"/>']
    for sx in (-hx + 0.004, hx - 0.004):
        for sy in (-hy + 0.004, hy - 0.004):
            out.append(f'<geom class="env_vis" type="cylinder" size="{_f(0.003, (deck_z - S.TUBE_RACK_BASE) / 2)}" '
                       f'pos="{_f(sx, sy, (deck_z + S.TUBE_RACK_BASE) / 2)}" material="rack_white"/>')
    # deck: bars along the row and between the tube holes
    for sy in (-hy + 0.003, hy - 0.003):
        out.append(f'<geom class="env_vis" type="box" size="{_f(hx, 0.003, 0.002)}" pos="{_f(0, sy, deck_z)}" material="rack_white"/>')
    for sx in (-hx + 0.003, -S.TUBE_RACK_PITCH / 2, S.TUBE_RACK_PITCH / 2, hx - 0.003):
        out.append(f'<geom class="env_vis" type="box" size="{_f(0.003, hy, 0.002)}" pos="{_f(sx, 0, deck_z)}" material="rack_white"/>')
    out.append(f'<geom class="env_vis" type="box" size="{_f(hx, 0.0012, 0.006)}" pos="{_f(0, -hy - 0.0012, deck_z - 0.004)}" '
               f'material="accent_purple"/>')
    for i in range(S.N_TUBES):
        name = f"tube_{i + 1}"
        tx, ty = S.tube_xy(i)
        out += [
            f'<geom name="{name}_wall" class="env_vis" type="mesh" mesh="tube_shell" pos="{_f(tx, ty, cone_top)}" material="pp_clear"/>',
            f'<geom class="env_vis" type="mesh" mesh="tube_cone_outer" pos="{_f(tx, ty, cone_top)}" material="pp_clear"/>',
            f'<geom name="{name}_cone" class="env" type="mesh" mesh="tube_cone_inner" pos="{_f(tx, ty, cone_top)}"/>',
            f'<site name="{name}_floor" pos="{_f(tx, ty, floor)}" size="0.002" group="4"/>',
        ]
        out += _liquid(name, spec, cfg.tube_volume_ul, tx, ty, floor, S.LIQUID_COLORS[cfg.tube_colors[i]])
        out += _ring_walls(tx, ty, cone_top, rim, spec.radius, S.TUBE_WALL)
    out.append('</body>')
    return out


def _dish(cfg: SceneConfig) -> list[str]:
    x, y, yaw = cfg.dish.pose()
    spec, t, floor = S.DISH, S.DISH_WALL, S.DISH_FLOOR_Z
    out = [f'<body name="dish" pos="{_f(x, y, 0)}" quat="{_f(*_yaw_quat(yaw))}">',
           f'<geom name="dish_bottom" class="env_both" type="cylinder" size="{_f(spec.radius + t, floor / 2)}" '
           f'pos="{_f(0, 0, floor / 2)}" material="glass"/>',
           f'<geom name="dish_wall" class="env_vis" type="mesh" mesh="dish_shell" pos="{_f(0, 0, floor)}" material="glass"/>',
           f'<geom class="env_vis" type="mesh" mesh="dish_trim" pos="0 0 0" material="accent_purple"/>',
           f'<site name="dish_floor" pos="{_f(0, 0, floor)}" size="0.002" group="4"/>']
    out += _liquid("dish", spec, cfg.dish_volume_ul, 0, 0, floor, S.LIQUID_COLORS[cfg.tube_colors[0]])
    out += _ring_walls(0, 0, floor, floor + spec.depth, spec.radius, t, n=16)
    out.append('</body>')
    return out


# ----------------------------------------------------------------------------------------------
# whole scene
# ----------------------------------------------------------------------------------------------
def build_scene_xml(cfg: SceneConfig | None = None) -> str:
    cfg = cfg or SceneConfig()
    tube = S.TUBE_50ML
    meshes = {
        "tube_cone_outer": _ring(tube.radius + S.TUBE_WALL, 0.0) + _ring(0.003, -tube.cone_height - 0.001),
        "tube_cone_inner": _ring(tube.radius, 0.0) + _ring(0.0005, -tube.cone_height),
        "liquid_cone": _ring(tube.radius - 0.0002, 0.0) + _ring(0.0003, -tube.cone_height + 0.0002),
    }
    mesh_xml = [f'<mesh name="{k}" vertex="{_f(*v)}"/>' for k, v in meshes.items()]
    mesh_xml.append(_shell_mesh("tube_shell", tube.radius + S.TUBE_WALL, tube.radius, tube.depth - tube.cone_height))
    mesh_xml.append(_shell_mesh("dish_shell", S.DISH.radius + S.DISH_WALL, S.DISH.radius, S.DISH.depth, n=48))
    r_out = S.DISH.radius + S.DISH_WALL
    mesh_xml.append(_shell_mesh("dish_trim", r_out + 0.0015, r_out + 0.0002, 0.004, n=48))

    ready = _f(*READY_Q)  # keyframe: arm in READY_Q, plunger at rest
    # "side" (policy camera) sits at the front-right corner of the bench, so the source rack and the
    # dish appear side by side; from a true side view they would line up behind each other.
    side_pos, overview_pos, top_pos = (0.85, -0.25, 0.45), (1.20, -0.12, 0.62), (0.22, 0.0, 0.95)
    side_cam = look_at(side_pos, (0.24, 0.0, 0.12))
    overview_cam = look_at(overview_pos, (0.14, 0.0, 0.29))
    top_cam = look_at(top_pos, (0.22, 0.0, 0.0), up=(1.0, 0.0, 0.0))

    xml = [
        '<mujoco model="pipette_transfer_scene">',
        '<!-- GENERATED by pipette_sim/scene_builder.py (python scripts/build_scene.py). Do not edit by hand. -->',
        '<include file="gen3_lite_pipette.xml"/>',
        '<option timestep="0.002" integrator="implicitfast"/>',
        '<statistic center="0.2 0 0.15" extent="0.9" meansize="0.03"/>',
        '<visual>',
        '  <headlight ambient="0.28 0.28 0.3" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>',
        '  <quality shadowsize="4096" offsamples="4"/>',
        '  <global offwidth="1920" offheight="1080" azimuth="180" elevation="-20"/>',
        '  <map znear="0.01" zfar="20"/>',
        '</visual>',
        '<default>',
        f'  <default class="env"><geom contype="{ENV}" conaffinity="{ARM | TOOL | TIP}" group="3" rgba="0.6 0.6 0.6 0.4"/></default>',
        f'  <default class="env_both"><geom contype="{ENV}" conaffinity="{ARM | TOOL | TIP}" group="2"/></default>',
        '  <default class="env_vis"><geom contype="0" conaffinity="0" group="2" density="0"/></default>',
        '  <default class="liquid"><geom type="cylinder" contype="0" conaffinity="0" group="1" density="0" material="liquid"/></default>',
        '</default>',
        '<asset>',
        '  <texture type="skybox" builtin="gradient" rgb1="0.82 0.85 0.88" rgb2="0.45 0.48 0.52" width="512" height="3072"/>',
        '  <texture name="floor" type="2d" builtin="checker" rgb1="0.55 0.56 0.58" rgb2="0.48 0.49 0.51" width="512" height="512"/>',
        '  <material name="floor" texture="floor" texrepeat="6 6" reflectance="0.05"/>',
        '  <material name="bench" rgba="0.21 0.23 0.26 1" specular="0.25" shininess="0.3"/>',
        '  <material name="steel" rgba="0.6 0.62 0.65 1" specular="0.6" shininess="0.6"/>',
        '  <material name="rack_white" rgba="0.92 0.92 0.94 1" specular="0.3"/>',
        '  <material name="accent_purple" rgba="0.62 0.52 0.86 1" specular="0.3"/>',
        '  <material name="pp_clear" rgba="0.90 0.92 0.95 0.28" specular="0.6" shininess="0.8"/>',
        '  <material name="glass" rgba="0.80 0.90 0.95 0.25" specular="0.8" shininess="0.9"/>',
        *[f'  {m}' for m in mesh_xml],
        '</asset>',
        '<worldbody>',
        '  <light name="key" type="spot" pos="0.9 -0.7 1.8" dir="-0.4 0.4 -1" cutoff="50" exponent="1" diffuse="0.65 0.65 0.62" specular="0.25 0.25 0.25" castshadow="true"/>',
        '  <light name="fill" type="directional" pos="-0.5 0.7 1.8" dir="0.3 -0.35 -1" diffuse="0.30 0.30 0.33" specular="0 0 0" castshadow="false"/>',
        '  <geom name="floor" class="env_both" type="plane" size="3 3 0.05" pos="0 0 -0.92" material="floor"/>',
        '  <geom name="bench_top" class="env_both" type="box" size="0.55 0.75 0.02" pos="0.15 0 -0.02" material="bench"/>',
        *[f'  <geom class="env_vis" type="box" size="0.025 0.025 0.44" pos="{_f(lx, ly, -0.48)}" material="steel"/>'
          for lx in (-0.36, 0.66) for ly in (-0.71, 0.71)],
        f'  <camera name="side" pos="{_f(*side_pos)}" xyaxes="{side_cam}" fovy="45" resolution="1280 720"/>',
        f'  <camera name="overview" pos="{_f(*overview_pos)}" xyaxes="{overview_cam}" fovy="45" resolution="1280 720"/>',
        f'  <camera name="top" pos="{_f(*top_pos)}" xyaxes="{top_cam}" fovy="50" resolution="1280 720"/>',
        *[f'  {line}' for line in _tube_rack(cfg)],
        *[f'  {line}' for line in _dish(cfg)],
        '</worldbody>',
        '<keyframe>',
        f'  <key name="ready" qpos="{ready} 0" ctrl="{ready} 0"/>',
        '</keyframe>',
        '</mujoco>',
    ]
    return "\n".join(xml) + "\n"


def write_scene(cfg: SceneConfig | None = None, path: Path | None = None) -> Path:
    path = path or MODELS_DIR / "scene.xml"
    path.write_text(build_scene_xml(cfg))
    return path
