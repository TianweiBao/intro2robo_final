"""Check stock Gen3 lite gripper motion and independent arm/tool commands, without rendering.

The pipette is a provisional rigid attachment. These checks do not establish a frictional grasp
or release behavior. Run after ``python scripts/build_scene.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pipette_sim import specs as S  # noqa: E402
from pipette_sim.sim import ENV, GRIPPER, PipetteSim  # noqa: E402


def tip_in_flange(sim: PipetteSim) -> np.ndarray:
    flange = sim.data.site("flange")
    return flange.xmat.reshape(3, 3).T @ (sim.tcp() - flange.xpos)


def main() -> int:
    results: list[bool] = []

    def check(name: str, good: bool, detail: str = "") -> None:
        results.append(bool(good))
        print(f"{'PASS' if good else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))

    sim = PipetteSim()
    try:
        m, d = sim.model, sim.data
        actuator_ids = [*sim.act_arm, sim.act_gripper, sim.act_plunger]
        check("six arm, one gripper and one plunger actuator are distinct",
              len(set(actuator_ids)) == 8 and m.nu == 8)
        check("stock gripper driver limits", np.allclose(sim.gripper_range, [S.GRIPPER_OPEN, S.GRIPPER_CLOSED]))
        check("ready keyframe restores every joint and actuator",
              np.array_equal(d.qpos, m.key_qpos[sim.key_ready])
              and np.array_equal(d.ctrl, m.key_ctrl[sim.key_ready]))
        check("ready gripper joints satisfy the URDF mimic relations",
              np.allclose(sim.gripper_q, S.gripper_joint_positions(sim.gripper_pos), atol=1e-7))

        # Each public command must preserve the commands owned by the other controllers.
        before = d.ctrl.copy()
        arm_target = sim.arm_q + np.array([0.002, 0, 0, 0, 0, 0])
        sim.set_arm_targets(arm_target)
        arm_mask = np.ones(m.nu, dtype=bool)
        arm_mask[sim.act_arm] = False
        check("arm commands leave gripper and plunger targets unchanged",
              np.array_equal(d.ctrl[arm_mask], before[arm_mask])
              and np.array_equal(d.ctrl[sim.act_arm], arm_target))
        before = d.ctrl.copy()
        sim.set_gripper(0.4)
        other = np.arange(m.nu) != sim.act_gripper
        check("gripper commands leave arm and plunger targets unchanged",
              np.array_equal(d.ctrl[other], before[other]) and sim.gripper_target == 0.4)
        before = d.ctrl.copy()
        sim.set_plunger(0.005)
        sim.step()
        other = np.arange(m.nu) != sim.act_plunger
        check("plunger advances independently of arm and gripper commands",
              np.array_equal(d.ctrl[other], before[other])
              and 0 < d.ctrl[sim.act_plunger] <= sim.plunger_target)
        sim.set_gripper(-100)
        low_ok = sim.gripper_target == sim.gripper_range[0]
        sim.set_gripper(100)
        check("gripper commands respect the stock driver limits",
              low_ok and sim.gripper_target == sim.gripper_range[1])

        # Sample the complete mechanism kinematically before testing actuator dynamics.
        # The attached pipette must remain fixed relative to the flange as fingers move.
        sim.reset()
        fixed_tip = tip_in_flange(sim)
        apertures = []
        fixed_error = 0.0
        mirror_error = 0.0
        for angle in np.linspace(*sim.gripper_range, 11):
            d.qpos[sim.gripper_qadr] = S.gripper_joint_positions(angle)
            mujoco.mj_forward(m, d)
            apertures.append(sim.gripper_aperture)
            fixed_error = max(fixed_error, float(np.linalg.norm(tip_in_flange(sim) - fixed_tip)))
            base = d.body("gripper_base_link")
            left, right = (d.site_xpos[sim.gripper_pad_sites] - base.xpos) @ base.xmat.reshape(3, 3)
            residual = left - right * np.array([1, -1, 1])
            mirror_error = max(mirror_error, float(np.max(np.abs(residual))))
        check("increasing driver angle closes the pads across the full travel",
              bool(np.all(np.diff(apertures) < 0)),
              f"open {apertures[0] * 1000:.1f} mm, closed {apertures[-1] * 1000:.1f} mm")
        check("pad geometry remains symmetric across the gripper center plane", mirror_error < 0.0002,
              f"largest mirror mismatch {mirror_error * 1000:.2g} mm")
        check("finger motion preserves the provisional rigid pipette attachment", fixed_error < 1e-8,
              f"largest tip shift in flange frame {fixed_error:.2g} m")

        sim.reset()
        ready_arm = sim.arm_q
        ready_gripper = sim.gripper_target
        seen = set()
        max_mimic_error = max_driver_error = max_arm_drift = max_tip_shift = 0.0
        for angle in (S.GRIPPER_OPEN, S.GRIPPER_CLOSED, ready_gripper):
            sim.set_gripper(angle)
            for _ in range(125):  # 2.5 s per endpoint at the default 50 Hz control rate
                sim.step()
                seen.update(sim.contacts())
                max_tip_shift = max(max_tip_shift, float(np.linalg.norm(tip_in_flange(sim) - fixed_tip)))
            max_mimic_error = max(max_mimic_error,
                                 float(np.max(np.abs(sim.gripper_q - S.gripper_joint_positions(sim.gripper_pos)))))
            max_driver_error = max(max_driver_error, abs(sim.gripper_pos - angle))
            max_arm_drift = max(max_arm_drift, float(np.max(np.abs(sim.arm_q - ready_arm))))
        check("gripper actuator opens, closes and returns to its ready target", max_driver_error < 0.003,
              f"largest settled driver error {max_driver_error:.2g} rad")
        check("both fingers follow the coupled mechanism symmetrically", max_mimic_error < 0.002,
              f"largest settled mimic error {max_mimic_error:.2g} rad")
        check("arm holds position during independent gripper motion", max_arm_drift < 0.002,
              f"largest settled drift {max_arm_drift:.2g} rad")
        check("rigid pipette pose is preserved during dynamic finger motion", max_tip_shift < 1e-8,
              f"largest tip shift in flange frame {max_tip_shift:.2g} m")
        check("no unintended contacts during the open-close sequence", not seen,
              ", ".join(sorted(seen)) or "none")
        state = sim.state()
        check("state exposes measured gripper joints, target and pad aperture",
              np.array_equal(state["gripper_q"], sim.gripper_q)
              and len(state["gripper_qd"]) == len(S.GRIPPER_JOINTS)
              and state["gripper_target"] == sim.gripper_target
              and state["gripper_aperture_m"] == sim.gripper_aperture)

        # Move a small environment probe into a finger in memory, checking that the new collision
        # group contributes to both readable contacts and the episode penalty. reset restores it.
        sim.reset()
        probe = m.geom("bench_top").id
        finger = m.geom("gripper_right_finger_dist_link_collision").id
        ready_qpos, ready_ctrl = d.qpos.copy(), d.ctrl.copy()
        outward = -d.body("gripper_base_link").xmat.reshape(3, 3)[:, 1]
        m.geom_pos[probe] = d.geom_xpos[finger] + 0.007 * outward
        m.geom_size[probe] = [0.003, 0.003, 0.003]
        # Changing a static geom requires rebuilding the collision bounds. mj_setConst uses
        # qpos0 internally, so restore the sampled pose before asking for contacts.
        mujoco.mj_setConst(m, d)
        d.qpos[:], d.ctrl[:] = ready_qpos, ready_ctrl
        mujoco.mj_forward(m, d)
        has_gripper_contact = any(t == {ENV, GRIPPER} for _, _, t in sim._contact_pairs())
        sim._check_contacts()
        check("gripper-environment contacts are reported and penalized",
              has_gripper_contact and any("gripper_" in pair for pair in sim.contacts())
              and sim.events.robot_collision_steps == 1)
        sim.reset()
        mujoco.mj_setConst(m, d)
        sim.reset()
    finally:
        sim.close()
    print(f"\n{sum(results)}/{len(results)} gripper checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
