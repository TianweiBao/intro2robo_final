"""Runtime wrapper around models/scene.xml.

MuJoCo simulates the arm, the pipette plunger and contacts. This class adds what MuJoCo does not
model: liquid moving between the tubes, the tip and the dish as the plunger moves.

Task: three tubes hold red, blue and green liquid. The target color selects the source tube, and
the destination is the dish.

Typical use:
    sim = PipetteSim()
    sim.reset(target_color="blue")
    sim.set_arm_targets(q)        # 6 joint angles for the arm's position actuators
    sim.set_plunger(0.010)        # plunger travel in m (0.010 = first stop = 1000 uL)
    sim.step()                    # one control step (control_dt seconds)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import specs as S
from .ik import ARM_JOINTS
from .liquid import Container, LiquidVisuals, mix_rgb

SCENE_XML = Path(__file__).resolve().parent.parent / "models" / "scene.xml"
ENV, ARM, TOOL, TIP = 1, 2, 4, 8

TUBES = tuple(f"tube_{i}" for i in range(1, S.N_TUBES + 1))
CONTAINERS = TUBES + ("dish",)
LABWARE_BODIES = ("tube_rack", "dish")
COLORS = tuple(S.LIQUID_COLORS)


@dataclass
class Events:
    """Episode counters, used by the checks now and for rewards later."""

    aspirated_ul: float = 0.0
    dispensed_ul: float = 0.0
    spilled_ul: float = 0.0
    wrong_source_ul: float = 0.0       # drawn from a tube that does not hold the target color
    robot_collision_steps: int = 0     # arm or pipette touching the bench or labware
    self_collision_steps: int = 0      # pipette or camera touching the arm
    tip_contact_steps: int = 0         # tip touching a tube wall, the dish or the bench
    log: list[str] = field(default_factory=list)


class PipetteSim:
    def __init__(self, xml_path: str | Path = SCENE_XML, control_dt: float = 0.02):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.control_dt = control_dt
        self.n_substeps = max(1, round(control_dt / m.opt.timestep))

        self.arm_qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.act_plunger = m.actuator("plunger").id
        self.tip_end_site = m.site("tip_end").id
        self.nozzle_site = m.site("nozzle").id
        self.tip_liquid_geoms = [m.geom(f"tip_liquid{k}").id for k in range(len(S.TIP_LIQUID_SEGMENTS))]
        self.key_ready = m.key("ready").id

        self.containers: dict[str, Container] = {}
        for name in CONTAINERS:
            g = m.geom(f"{name}_liquid").id
            self.containers[name] = Container(
                name=name, spec=S.DISH if name == "dish" else S.TUBE_50ML, body_id=m.geom_bodyid[g],
                floor_site=m.site(f"{name}_floor").id, liquid_geom=g,
                cone_geom=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_liquid_cone"),
                base_radius=float(m.geom_size[g, 0]))
        self.visuals = LiquidVisuals(m)

        # Values that randomization may change; reset() restores them.
        self._nominal = {k: getattr(m, k).copy() for k in (
            "body_pos", "body_quat", "geom_rgba", "geom_size", "geom_pos", "mat_rgba",
            "light_pos", "light_dir", "light_diffuse", "cam_pos", "cam_quat", "cam_fovy")}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.reset()

    # ------------------------------------------------------------------------------------------
    # episode setup
    # ------------------------------------------------------------------------------------------
    def reset(self, tube_volumes=None, tube_colors=None, dish_volume: float = 0.0,
              target_color: str | None = None) -> None:
        """Restore the nominal scene, put the arm in the ready pose and refill the containers.

        tube_volumes: one value for all tubes or one per tube (uL), default 30 mL.
        tube_colors:  colors of tube_1..tube_3, default ("red", "blue", "green").
        target_color: the liquid to transfer; it selects the source tube. Default: tube_1's color.
        """
        m, d = self.model, self.data
        for k, v in self._nominal.items():
            getattr(m, k)[:] = v
        mujoco.mj_resetDataKeyframe(m, d, self.key_ready)

        colors = tuple(tube_colors or S.DEFAULT_TUBE_COLORS)
        assert sorted(colors) == sorted(COLORS), f"tube colors must be a permutation of {COLORS}"
        self.tube_colors = dict(zip(TUBES, colors))
        self.target_color = target_color or colors[0]
        assert self.target_color in COLORS
        vols = np.broadcast_to(np.asarray(30000.0 if tube_volumes is None else tube_volumes, float), (len(TUBES),))
        for name, c in self.containers.items():
            if name == "dish":
                c.volume, c.rgb = float(dish_volume), np.array(S.LIQUID_COLORS[self.target_color])
            else:
                c.volume, c.rgb = float(vols[TUBES.index(name)]), np.array(S.LIQUID_COLORS[self.tube_colors[name]])
            self.visuals.update_container(c)

        self.tip_volume = 0.0
        self.tip_rgb = np.zeros(3)
        self.visuals.update_tip(self.tip_liquid_geoms, 0.0, self.tip_rgb)
        self.plunger_target = 0.0
        self.plunger_pos = 0.0          # commanded plunger position after the actuator's speed limit
        self._plunger_prev = 0.0
        self.events = Events()
        mujoco.mj_forward(m, d)

    def tube_with(self, color: str) -> str:
        return next(t for t, c in self.tube_colors.items() if c == color)

    @property
    def source_tube(self) -> str:
        """The tube holding the target color (privileged: a vision policy must find it in the images)."""
        return self.tube_with(self.target_color)

    # ------------------------------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------------------------------
    @property
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    def set_arm_targets(self, q) -> None:
        self.data.ctrl[:6] = q

    def set_plunger(self, travel: float) -> None:
        """Plunger target in meters: 0 = rest, PLUNGER_FIRST_STOP = 1000 uL, PLUNGER_BLOWOUT = second stop.

        The linear actuator moves toward the target at up to PLUNGER_MAX_SPEED. Liquid follows this
        commanded position; the simulated plunger joint only tracks it visually.
        """
        self.plunger_target = float(np.clip(travel, 0.0, S.PLUNGER_BLOWOUT))

    def step(self, n: int = 1) -> None:
        """Advance n control steps (physics substeps, then liquid and contact bookkeeping)."""
        max_dx = S.PLUNGER_MAX_SPEED * self.control_dt
        for _ in range(n):
            self.plunger_pos += float(np.clip(self.plunger_target - self.plunger_pos, -max_dx, max_dx))
            self.data.ctrl[self.act_plunger] = self.plunger_pos
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
            if self._move_liquid():
                mujoco.mj_kinematics(self.model, self.data)
            self._check_contacts()

    # ------------------------------------------------------------------------------------------
    # geometry queries
    # ------------------------------------------------------------------------------------------
    def tcp(self) -> np.ndarray:
        """Controlled point: the end of the pipette tip."""
        return self.data.site_xpos[self.tip_end_site].copy()

    def tool_frame(self) -> np.ndarray:
        """Pipette frame rotation: z down the pipette, x from the flange toward the pipette."""
        return self.data.site_xmat[self.tip_end_site].reshape(3, 3).copy()

    def container_frame(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World position of the container floor and the container rotation matrix (z up)."""
        sid = self.containers[name].floor_site
        return self.data.site_xpos[sid].copy(), self.data.site_xmat[sid].reshape(3, 3).copy()

    def surface_z(self, name: str) -> float:
        c = self.containers[name]
        floor, _ = self.container_frame(name)
        return float(floor[2] + c.spec.height_of_volume(c.volume))

    def rim_z(self, name: str) -> float:
        floor, _ = self.container_frame(name)
        return float(floor[2] + self.containers[name].spec.depth)

    def container_at(self, point: np.ndarray) -> tuple[Container | None, float]:
        """Container whose opening contains `point` (below its rim); returns (container, height above floor)."""
        for c in self.containers.values():
            floor, R = self.container_frame(c.name)
            local = R.T @ (point - floor)
            if -0.003 <= local[2] <= c.spec.depth and c.spec.inside_xy(local[0], local[1]):
                return c, float(local[2])
        return None, 0.0

    # ------------------------------------------------------------------------------------------
    # liquid and contact bookkeeping (once per control step)
    # ------------------------------------------------------------------------------------------
    def _move_liquid(self) -> bool:
        d = self.data
        dx, self._plunger_prev = self.plunger_pos - self._plunger_prev, self.plunger_pos
        if abs(dx) < 1e-9:
            return False
        end = d.site_xpos[self.tip_end_site]
        c, h_end = self.container_at(end)
        if dx > 0:  # plunger down: liquid leaves the tip first, then air
            out = min(dx * S.UL_PER_M_PLUNGER, self.tip_volume)
            if out <= 0:
                return False
            self.tip_volume -= out
            if c is not None:
                self.events.spilled_ul += c.add(out, self.tip_rgb)
                self.events.dispensed_ul += out
                self.visuals.update_container(c)
            else:
                self.events.spilled_ul += out
                self.events.log.append(f"t={d.time:.2f} spilled {out:.0f} uL outside any container")
        else:  # plunger up: draws liquid only while the tip end is below the surface
            if c is None:
                return False
            h_surf = c.spec.height_of_volume(c.volume)
            if h_end > h_surf - S.MIN_IMMERSION:
                return False
            available = c.volume - c.spec.volume_at_height(max(h_end, 0.0))
            room = 1.3 * S.PIPETTE_VOLUME_UL - self.tip_volume
            take = max(0.0, min(-dx * S.UL_PER_M_PLUNGER, available, room))
            if take <= 0:
                return False
            self.tip_rgb = mix_rgb(self.tip_rgb, self.tip_volume, c.rgb, take)
            self.tip_volume += take
            c.volume -= take
            self.events.aspirated_ul += take
            if c.name in TUBES and c.name != self.source_tube:
                self.events.wrong_source_ul += take
            self.visuals.update_container(c)
        self.visuals.update_tip(self.tip_liquid_geoms, self.tip_volume, self.tip_rgb)
        return True

    def _contact_pairs(self):
        m, d = self.model, self.data
        for k in range(d.ncon):
            g1, g2 = d.contact[k].geom1, d.contact[k].geom2
            yield g1, g2, {int(m.geom_contype[g1]), int(m.geom_contype[g2])}

    def _check_contacts(self) -> None:
        robot = selfc = tip = False
        for _, _, t in self._contact_pairs():
            robot |= bool(t & {ARM, TOOL}) and ENV in t
            selfc |= ARM in t and bool(t & {TOOL, TIP})
            tip |= TIP in t and ENV in t
        self.events.robot_collision_steps += int(robot)
        self.events.self_collision_steps += int(selfc)
        self.events.tip_contact_steps += int(tip)

    def contacts(self) -> list[str]:
        """Readable list of current robot/pipette/tip contacts (robot-labware, self, tip-labware)."""
        m = self.model
        name = lambda g: m.geom(g).name or m.body(m.geom_bodyid[g]).name
        return [f"{name(g1)}<->{name(g2)}" for g1, g2, t in self._contact_pairs()
                if (t & {ARM, TOOL} and ENV in t) or (ARM in t and t & {TOOL, TIP}) or (TIP in t and ENV in t)]

    # ------------------------------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------------------------------
    def state(self) -> dict:
        """Privileged simulator state (for scripted experts, rewards and the PPO critic)."""
        d = self.data
        return {
            "time": float(d.time),
            "arm_q": self.arm_q,
            "arm_qd": d.qvel[self.arm_dadr].copy(),
            "tcp": self.tcp(),
            "tool_frame": self.tool_frame(),
            "plunger": self.plunger_pos,
            "tip_volume_ul": self.tip_volume,
            "volumes_ul": {k: c.volume for k, c in self.containers.items()},
            "tube_colors": dict(self.tube_colors),
            "target_color": self.target_color,
            "source_tube": self.source_tube,
        }

    def render(self, camera: str, width: int = 640, height: int = 360, depth: bool = False) -> np.ndarray:
        """RGB (uint8, HxWx3) or depth in meters (float32, HxW) from a named camera."""
        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height, width)
        r = self._renderers[key]
        if depth:
            r.enable_depth_rendering()
        r.update_scene(self.data, camera=camera)
        img = r.render()
        if depth:
            r.disable_depth_rendering()
        return img

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
