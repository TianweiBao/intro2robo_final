"""Evaluate RGB basin detection and tube association across randomized MuJoCo scenes.

Examples:
    python scripts/evaluate_sim_vision.py --episodes 100 --seed 0
    python scripts/evaluate_sim_vision.py --episodes 500 --seed 100 --no-save-failures
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from pipette_sim.basin_vision import annotate_observations  # noqa: E402
from pipette_sim.sim import PipetteSim  # noqa: E402
from pipette_sim.vision_evaluation import evaluate_episode, summarize_results  # noqa: E402


def failure_image(rgb, observations, association, result):
    """Create an annotated image plus a text panel containing truth and prediction metadata."""
    tube_callouts = [] if not association.valid else [
        (item.observation, f"{item.tube_name.upper()} / {item.color.upper()}")
        for item in association.associations
    ]
    image = annotate_observations(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), observations, tube_callouts)
    panel_height = 86
    image = cv2.copyMakeBorder(image, 0, panel_height, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    lines = (
        f"Episode {result.episode_index}  seed={result.episode_seed}  result={result.classification}",
        f"Truth: {result.truth_tube_colors}",
        f"Predicted: {result.predicted_tube_colors or 'none'}",
        f"Reason: {result.failure_reason or 'n/a'}",
    )
    y0 = image.shape[0] - panel_height + 19
    for index, line in enumerate(lines):
        cv2.putText(image, line, (10, y0 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return image


def write_csv(path: Path, results) -> None:
    fields = ("episode_index", "episode_seed", "classification", "association_valid", "association_score",
              "truth_tube_colors", "predicted_tube_colors", "detected_colors", "failure_reason")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "episode_index": result.episode_index,
                "episode_seed": result.episode_seed,
                "classification": result.classification,
                "association_valid": result.association_valid,
                "association_score": result.association_score,
                "truth_tube_colors": json.dumps(result.truth_tube_colors, sort_keys=True),
                "predicted_tube_colors": json.dumps(result.predicted_tube_colors, sort_keys=True),
                "detected_colors": ";".join(result.detected_colors),
                "failure_reason": result.failure_reason or "",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=100, help="number of independently seeded scenes")
    parser.add_argument("--seed", type=int, default=0, help="first episode seed; later episodes increment it")
    parser.add_argument("--width", type=int, default=640, help="side-camera render width")
    parser.add_argument("--height", type=int, default=360, help="side-camera render height")
    parser.add_argument("--min-area", type=float, default=20.0, help="minimum colored region area in pixels")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="directory for reports and failures")
    parser.add_argument("--no-save-failures", action="store_true", help="do not save annotated failure PNG/JSON artifacts")
    parser.add_argument("--max-saved-failures", type=int, default=20, help="maximum failure artifact pairs to save")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_saved_failures < 0:
        parser.error("--max-saved-failures must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_dir = args.output_dir / "vision_eval_failures"
    if not args.no_save_failures:
        failure_dir.mkdir(parents=True, exist_ok=True)

    sim, results, saved_failures = PipetteSim(), [], 0
    try:
        for index in range(args.episodes):
            episode_seed = args.seed + index
            result, rgb, observations, association = evaluate_episode(
                sim, index, episode_seed, args.width, args.height, args.min_area)
            results.append(result)
            if result.classification != "correct" and not args.no_save_failures and saved_failures < args.max_saved_failures:
                stem = failure_dir / f"episode_{index:04d}_{result.classification}"
                if not cv2.imwrite(str(stem.with_suffix(".png")), failure_image(rgb, observations, association, result)):
                    raise RuntimeError(f"Could not write failure image {stem.with_suffix('.png')}")
                stem.with_suffix(".json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
                saved_failures += 1
            if (index + 1) % 25 == 0 or index + 1 == args.episodes:
                print(f"Evaluated {index + 1}/{args.episodes} scenes")
    finally:
        sim.close()

    summary = summarize_results(results)
    summary_path, csv_path = args.output_dir / "vision_eval_summary.md", args.output_dir / "vision_eval_results.csv"
    summary_path.write_text(summary.as_markdown(), encoding="utf-8")
    write_csv(csv_path, results)
    print(summary.as_markdown())
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {csv_path}")
    if not args.no_save_failures:
        print(f"Saved failure artifact pairs: {saved_failures}")


if __name__ == "__main__":
    main()
