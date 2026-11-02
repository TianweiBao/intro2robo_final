"""Phase 1 scene checks: model sanity, settling, reachability, the scripted three-phase task for
each color (with video), scripted tasks in randomized scenes, and speed. Writes
outputs/check_report.md plus images and a video.

    conda activate mujoco
    python scripts/check_scene.py            # full run (~30 s)
    python scripts/check_scene.py --no-video # skip the video
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must be set before importing mujoco
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imageio  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from pipette_sim import scripted  # noqa: E402
from pipette_sim import specs as S  # noqa: E402
from pipette_sim.ik import VerticalIK  # noqa: E402
from pipette_sim.motion import MotionPrimitives  # noqa: E402
from pipette_sim.randomization import FOOTPRINT_HALF, _rects_overlap, randomize  # noqa: E402
from pipette_sim.sim import COLORS, TUBES, PipetteSim  # noqa: E402

OUT = ROOT / "outputs"


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.results: list[tuple[str, bool]] = []

    def section(self, title: str) -> None:
        self.add(f"\n## {title}\n")

    def add(self, text: str) -> None:
        print(text)
        self.lines.append(text)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, ok))
        self.add(f"- {'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))

    def write(self, path: Path) -> None:
        n_ok = sum(ok for _, ok in self.results)
        head = [f"# Phase 1 scene check: {n_ok}/{len(self.results)} checks passed\n"]
        path.write_text("\n".join(head + self.lines) + "\n")


def depth_to_rgb(depth: np.ndarray, near: float = 0.05, far: float = 0.45) -> np.ndarray:
    """Depth (m) to a blue (near) -> yellow (far) false-color image."""
    t = np.clip((depth - near) / (far - near), 0, 1)
    stops = np.array([[0.05, 0.03, 0.45], [0.10, 0.45, 0.85], [0.30, 0.75, 0.45], [0.95, 0.85, 0.20], [0.95, 0.95, 0.85]])
    x = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(t, x, stops[:, c]) for c in range(3)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow without scalable default font
        return ImageFont.load_default()


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    dr.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def frame_grid(sim: PipetteSim, phase: str, task: str, w: int = 640, h: int = 360) -> np.ndarray:
    tiles = [label(sim.render("overview", w, h), "overview (documentation)"),
             label(sim.render("side", w, h), "side camera (policy input)"),
             label(sim.render("wrist", w, h), "wrist D405 RGB (policy input)"),
             label(depth_to_rgb(sim.render("wrist", w, h, depth=True)), "wrist D405 depth")]
    grid = Image.fromarray(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    dr = ImageDraw.Draw(grid)
    banner = f"{task}    {phase}" if phase else task
    dr.rectangle([w - 330, h - 26, w + 330, h + 26], fill=(70, 50, 120))
    dr.text((w, h), banner, fill=(255, 255, 255), font=_font(22), anchor="mm")
    s = sim.state()
    dr.text((10, h - 22), f"t={s['time']:5.1f}s  tip {s['tip_volume_ul']:6.1f} uL  dish {s['volumes_ul']['dish']:6.1f} uL",
            fill=(255, 230, 120))
    return np.asarray(grid)


def contacts_of(sim: PipetteSim) -> list[str]:
    return sim.contacts()


# ----------------------------------------------------------------------------------------------
def check_model(sim: PipetteSim, rep: Report) -> None:
    m = sim.model
    rep.section("1. Model")
    robot_mass = m.body_subtreemass[m.body("base_link").id]
    tool_mass = m.body_subtreemass[m.body("end_effector_link").id]
    rep.add(f"- nq={m.nq}, nv={m.nv}, actuators={m.nu}, bodies={m.nbody}, geoms={m.ngeom}, "
            f"cameras={[m.cam(i).name for i in range(m.ncam)]}")
    rep.add(f"- timestep {m.opt.timestep * 1e3:.1f} ms, control step {sim.control_dt * 1e3:.0f} ms "
            f"({sim.n_substeps} substeps), integrator implicitfast")
    rep.add(f"- robot mass {robot_mass:.3f} kg, of which tool {tool_mass:.3f} kg "
            f"(adapter, plate, actuator, manual pipette, tip, D405)")
    rep.check("tool within the Gen3 lite payload (0.5 kg)", tool_mass < 0.5, f"{tool_mass:.3f} kg")
    rep.add(f"- tubes (left to right): {sim.tube_colors}; destination: dish")


def check_settle(sim: PipetteSim, rep: Report) -> None:
    rep.section("2. Settling at the ready pose (2 s, no commands)")
    sim.reset()
    q0 = sim.arm_q
    seen = set()
    for _ in range(100):
        sim.step()
        seen.update(contacts_of(sim))
    rep.check("arm holds the ready pose", np.abs(sim.arm_q - q0).max() < 2e-3,
              f"max joint drift {np.degrees(np.abs(sim.arm_q - q0).max()):.3f} deg")
    rep.check("no contacts at rest", not seen, ", ".join(sorted(seen)) or "none")


def check_reachability(sim: PipetteSim, rep: Report) -> None:
    rep.section("3. Reachability: IK with the pipette vertical + collision check at each pose")
    sim.reset()
    m, d = sim.model, sim.data
    ik = VerticalIK(m)
    targets: list[tuple[str, np.ndarray]] = [("ready", scripted.READY_TCP)]
    for t in TUBES:
        f, _ = sim.container_frame(t)
        targets += [(f"{t} above (travel)", [f[0], f[1], scripted.TRAVEL_Z]),
                    (f"{t} aspirate @30 mL", [f[0], f[1], sim.surface_z(t) - scripted.IMMERSION]),
                    (f"{t} deepest (cone top)", [f[0], f[1], f[2] + S.TUBE_50ML.cone_height + 0.002])]
    f, _ = sim.container_frame("dish")
    targets += [("dish above (travel)", [f[0], f[1], scripted.TRAVEL_Z]),
                ("dish dispense", [f[0], f[1], f[2] + scripted.DISPENSE_HEIGHT]),
                ("dish near wall", [f[0] + 0.02, f[1], f[2] + scripted.DISPENSE_HEIGHT])]
    rows, failures, min_margin = [], [], np.inf
    for name, target in targets:
        q, ok = ik.solve_robust(np.asarray(target, float))
        margin = float(np.degrees(np.min(np.minimum(q - ik.lo, ik.hi - q))))
        d.qpos[sim.arm_qadr] = q
        mujoco.mj_forward(m, d)
        contacts = contacts_of(sim)
        if ok:
            min_margin = min(min_margin, margin)
        if not ok or contacts:
            failures.append(name)
        rows.append(f"| {name} | {'yes' if ok else 'NO'} | {margin:.0f} | {', '.join(contacts) or '-'} |")
    rep.add("| target (tip end) | IK | min joint-limit margin (deg) | contacts |\n|---|---|---|---|")
    for r in rows:
        rep.add(r)
    rep.check(f"all {len(targets)} working poses reachable without collisions", not failures,
              ", ".join(failures) or f"smallest joint-limit margin {min_margin:.0f} deg")
    sim.reset()


def check_tasks(sim: PipetteSim, rep: Report, video: bool) -> None:
    rep.section("4. Scripted three-phase task for each color (Aspirate -> Transfer -> Dispense)")
    frames: list[np.ndarray] = []
    phase = {"name": ""}
    counter = {"k": 0}
    task = {"text": ""}

    def on_step():
        counter["k"] += 1
        if video and counter["k"] % 4 == 0:  # 50 Hz control, every 4th step at 25 fps = 2x speed
            frames.append(frame_grid(sim, phase["name"], task["text"]))

    def on_phase(name: str) -> None:
        phase["name"] = "" if name == "return" else name

    results, ok_all, wall, sim_time = [], True, 0.0, 0.0
    for color in COLORS:
        sim.reset(target_color=color)
        mp = MotionPrimitives(sim, on_step)
        task["text"] = f"Task: move the {color.upper()} liquid to the dish"
        phase["name"] = ""
        t0 = time.time()
        r = scripted.run_task(mp, on_phase=on_phase)
        wall += time.time() - t0
        sim_time += sim.data.time
        ev = sim.events
        dish_rgb = sim.containers["dish"].rgb
        good = (r["source"] == sim.tube_with(color) and abs(r["dest_change_ul"] - 1000) < 1
                and abs(r["source_change_ul"] + 1000) < 1 and ev.spilled_ul < 1e-6 and ev.wrong_source_ul < 1e-6
                and ev.robot_collision_steps == 0 and ev.self_collision_steps == 0 and ev.tip_contact_steps == 0
                and np.allclose(dish_rgb, S.LIQUID_COLORS[color], atol=1e-6))
        ok_all &= good
        rep.add(f"- {color:5s}: source {r['source']} (expected {sim.tube_with(color)}), aspirated {r['aspirated_ul']} uL, "
                f"dish {r['dest_change_ul']:+.1f} uL, spilled {ev.spilled_ul:.1f} uL, collisions robot/self/tip "
                f"{ev.robot_collision_steps}/{ev.self_collision_steps}/{ev.tip_contact_steps} steps, "
                f"{sim.data.time:.1f} s")
        if video:
            frames += [frames[-1]] * 25 if frames else []  # 1 s pause between tasks
        results.append(r)
        if color == "blue":
            iio.imwrite(OUT / "after_blue_task_side.png", sim.render("side", 1280, 720))
    rep.check("each color is taken from the right tube and 1000 uL reaches the dish, with no spills or contacts", ok_all)
    rep.add(f"- simulated {sim_time:.1f} s in {wall:.1f} s wall time ({'with' if video else 'without'} video rendering)")
    if video and frames:
        with imageio.get_writer(OUT / "task_demo.mp4", fps=25, codec="libx264", quality=7, macro_block_size=8) as w:
            for f in frames:
                w.append_data(f)
        rep.add(f"- video: outputs/task_demo.mp4 ({len(frames) / 25:.0f} s, 2x speed)")


def check_randomization(rep: Report, n_scenes: int = 20, n_tasks: int = 6) -> None:
    rep.section(f"5. Domain randomization ({n_scenes} random scenes, scripted task in {n_tasks} of them)")
    sim = PipetteSim()
    ik = VerticalIK(sim.model)
    rng = np.random.default_rng(0)
    overlaps = unreachable = contact_scenes = task_failures = 0
    gallery = []
    for k in range(n_scenes):
        info = randomize(sim, rng)
        lab = info["labware"]
        (a, pa), (b, pb) = lab.items()
        overlaps += _rects_overlap(pa[0], FOOTPRINT_HALF[a], np.radians(pa[1]), pb[0], FOOTPRINT_HALF[b],
                                   np.radians(pb[1]), 0.0)
        sim.step(25)
        contact_scenes += bool(contacts_of(sim))
        for c in TUBES + ("dish",):
            f, _ = sim.container_frame(c)
            z = sim.surface_z(c) - 0.003 if c != "dish" else f[2] + scripted.DISPENSE_HEIGHT
            unreachable += not ik.solve_robust(np.array([f[0], f[1], z]))[1]
        if k < 4:
            gallery.append(np.hstack([sim.render("side", 480, 270), sim.render("wrist", 480, 270)]))
        if k < n_tasks:
            r = scripted.run_task(MotionPrimitives(sim))
            ev = sim.events
            ok = (r["source"] == sim.tube_with(info["target_color"]) and abs(r["dest_change_ul"] - 1000) < 1
                  and ev.spilled_ul < 1e-6 and ev.robot_collision_steps == 0 and ev.self_collision_steps == 0)
            task_failures += not ok
            rep.add(f"- scene {k}: tubes {info['tube_colors']}, target {info['target_color']} -> {r['source']}, "
                    f"dish {r['dest_change_ul']:+.1f} uL {'OK' if ok else 'FAILED'}")
    rep.check("labware never overlaps", overlaps == 0, f"{overlaps} scenes with overlaps")
    rep.check("no contacts after reset", contact_scenes == 0, f"{contact_scenes} scenes")
    rep.check("all tubes and the dish reachable in every scene", unreachable == 0,
              f"{unreachable} unreachable of {n_scenes * 4}")
    rep.check("scripted task succeeds with shuffled colors and moved labware", task_failures == 0,
              f"{n_tasks - task_failures}/{n_tasks}")
    iio.imwrite(OUT / "randomization_examples.png", np.vstack(gallery))
    rep.add("- examples (side | wrist): outputs/randomization_examples.png")
    sim.close()


def check_speed(rep: Report) -> None:
    rep.section("6. Speed (single process)")
    sim = PipetteSim()
    rng = np.random.default_rng(1)
    q0 = sim.arm_q
    n = 500
    t0 = time.time()
    for k in range(n):
        if k % 25 == 0:
            sim.set_arm_targets(q0 + rng.uniform(-0.05, 0.05, 6))
        sim.step()
    phys = n / (time.time() - t0)
    rep.add(f"- physics + liquid bookkeeping: {phys:.0f} control steps/s ({phys * sim.control_dt:.0f}x real time)")
    for w, h in ((128, 72), (224, 126)):
        sim.render("wrist", w, h)
        t0 = time.time()
        for _ in range(100):
            sim.step()
            sim.render("wrist", w, h)
            sim.render("side", w, h)
        rep.add(f"- physics + wrist & side RGB at {w}x{h}: {100 / (time.time() - t0):.0f} control steps/s")
    sim.close()


def render_stills(sim: PipetteSim) -> None:
    sim.reset()
    for cam in ("overview", "side", "top", "wrist"):
        iio.imwrite(OUT / f"view_{cam}.png", sim.render(cam, 1280, 720))
    iio.imwrite(OUT / "view_wrist_depth.png", depth_to_rgb(sim.render("wrist", 1280, 720, depth=True)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-video", action="store_true", help="skip the task video")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rep = Report()
    sim = PipetteSim()
    try:
        check_model(sim, rep)
        render_stills(sim)
        check_settle(sim, rep)
        check_reachability(sim, rep)
        check_tasks(sim, rep, video=not args.no_video)
    finally:
        sim.close()
    check_randomization(rep)
    check_speed(rep)
    rep.write(OUT / "check_report.md")
    n_ok = sum(ok for _, ok in rep.results)
    print(f"\n{n_ok}/{len(rep.results)} checks passed -> outputs/check_report.md")
    return 0 if n_ok == len(rep.results) else 1


if __name__ == "__main__":
    sys.exit(main())
