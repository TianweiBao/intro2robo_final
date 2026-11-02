"""Interactive MuJoCo viewer for the scene. Needs a display: run it from the lab machine's desktop,
or over SSH with DISPLAY=:1 to show it on the lab monitor.

    python scripts/view_scene.py                        # browse the scene at the ready pose
    python scripts/view_scene.py --demo                 # move red, then blue, then green to the dish
    python scripts/view_scene.py --demo --colors blue --speed 2

Viewer tips: double-click a body to select it, Ctrl+right-drag to push it, Tab hides the side panels,
the "Rendering" panel toggles geom groups (group 3 = collision shapes, group 4 = sites).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from pipette_sim import scripted  # noqa: E402
from pipette_sim.motion import MotionPrimitives  # noqa: E402
from pipette_sim.sim import PipetteSim  # noqa: E402


def run_demo(sim: PipetteSim, sync, is_running, colors=("red", "blue", "green"), speed: float = 1.0) -> None:
    """Run the three-phase task for each color (fresh scene each time), calling sync() after every
    control step at `speed` x real time."""
    for color in colors:
        sim.reset(target_color=color)
        t_wall0, t_sim0 = time.time(), sim.data.time

        def on_step():
            if not is_running():
                raise KeyboardInterrupt
            sync()
            lag = (sim.data.time - t_sim0) / speed - (time.time() - t_wall0)
            if lag > 0:
                time.sleep(lag)

        print(f"task: move the {color} liquid to the dish")
        result = scripted.run_task(MotionPrimitives(sim, on_step), on_phase=lambda name: print(f"  {name}"))
        print(f"  -> {result}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="run the scripted three-phase task")
    ap.add_argument("--colors", nargs="+", default=["red", "blue", "green"], choices=["red", "blue", "green"],
                    help="which liquids to transfer in the demo, in order")
    ap.add_argument("--speed", type=float, default=1.0, help="demo playback speed (x real time)")
    args = ap.parse_args()

    import mujoco.viewer  # imported here so the rest of the module works without a display

    sim = PipetteSim()
    if not args.demo:
        mujoco.viewer.launch(sim.model, sim.data)
        return
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = [0.15, 0.0, 0.25]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.3, 185, -18
        try:
            run_demo(sim, viewer.sync, viewer.is_running, args.colors, args.speed)
        except KeyboardInterrupt:
            return
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
