"""Run HSV basin perception on MuJoCo-rendered side or wrist camera images.

Examples:
    python scripts/sim_basin_vision_demo.py
    python scripts/sim_basin_vision_demo.py --randomize --seed 0 --show
    python scripts/sim_basin_vision_demo.py --camera wrist --save outputs/wrist_vision.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pipette_sim.basin_vision import annotate_observations, observe_rgb_frame, observed_status  # noqa: E402
from pipette_sim.randomization import randomize  # noqa: E402
from pipette_sim.sim import PipetteSim  # noqa: E402
from pipette_sim.tube_association import associate_tubes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", choices=("side", "wrist"), default="side", help="MuJoCo camera to render")
    parser.add_argument("--randomize", action="store_true", help="apply one domain-randomized scene")
    parser.add_argument("--seed", type=int, default=0, help="randomization seed")
    parser.add_argument("--width", type=int, default=640, help="render width in pixels")
    parser.add_argument("--height", type=int, default=360, help="render height in pixels")
    parser.add_argument("--min-area", type=float, default=20.0,
                        help="minimum colored region area in rendered pixels (default: 20)")
    parser.add_argument("--save", type=Path, help="annotated PNG path (default: outputs/sim_basin_vision_<camera>.png)")
    parser.add_argument("--show", action="store_true", help="also open an OpenCV window")
    args = parser.parse_args()

    sim = PipetteSim()
    try:
        if args.randomize:
            info = randomize(sim, np.random.default_rng(args.seed))
            print(f"Randomized scene (seed={args.seed}): {info['tube_colors']}")
        else:
            print(f"Nominal scene: {tuple(sim.tube_colors[tube] for tube in sim.tube_colors)}")

        rgb = sim.render(args.camera, args.width, args.height)
        observations = observe_rgb_frame(rgb, camera=args.camera, timestamp_s=sim.data.time,
                                         min_area_px=args.min_area)
        association = associate_tubes(observations) if args.camera == "side" else None
        tube_callouts = [] if association is None or not association.valid else [
            (item.observation, f"{item.tube_name.upper()} / {item.color.upper()}")
            for item in association.associations
        ]
        # annotate_observations uses OpenCV BGR, so convert the MuJoCo RGB image explicitly.
        annotated_bgr = annotate_observations(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), observations, tube_callouts)
        output = args.save or ROOT / "outputs" / f"sim_basin_vision_{args.camera}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), annotated_bgr):
            raise RuntimeError(f"Could not write {output}")

        print(f"Expected tube colors: {sim.tube_colors}")
        print(f"Detected basins: {observed_status(observations)}")
        for observation in observations:
            print(f"  {observation.color.upper():5} center={observation.image_center_px} "
                  f"area={observation.area_px:.0f}px confidence={observation.confidence:.2f}")
        if association is None:
            print("Tube association: side camera required (wrist image is a close-up view).")
        elif not association.valid:
            print(f"Tube association: unavailable ({association.reason})")
        else:
            print(f"Tube association score: {association.score:.2f}")
            for item in association.associations:
                print(f"  {item.tube_name}: {item.color.upper()} center={item.image_center_px} "
                      f"confidence={item.confidence:.2f}")
        print(f"Saved annotated image: {output}")

        if args.show:
            cv2.imshow(f"MuJoCo {args.camera} basin vision", annotated_bgr)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    finally:
        sim.close()


if __name__ == "__main__":
    main()
