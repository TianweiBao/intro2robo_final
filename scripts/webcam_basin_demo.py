"""Live laptop-webcam HSV color-basin identifier.

Example:
    python scripts/webcam_basin_demo.py --camera 0 --min-area 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from pipette_sim.basin_vision import annotate_observations, observe_basins  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--min-area", type=float, default=500.0, help="minimum colored contour area in pixels")
    parser.add_argument("--width", type=int, help="requested capture width in pixels")
    parser.add_argument("--height", type=int, help="requested capture height in pixels")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try another index with --camera.")

    window = "RGB basin identifier"
    print("Webcam demo running. Show red, green, or blue objects to the camera; press q or Esc to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam")
            observations = observe_basins(frame, camera="webcam", timestamp_s=time.monotonic(),
                                          min_area_px=args.min_area)
            cv2.imshow(window, annotate_observations(frame, observations))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
