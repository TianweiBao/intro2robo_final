"""Reusable evaluation primitives for simulated RGB basin perception.

The evaluator uses simulator truth only after perception has produced its prediction. This makes
the same metrics useful for later regression checks without leaking privileged state into the
perception or control path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .basin_vision import BasinObservation, observe_rgb_frame
from .randomization import randomize
from .sim import COLORS, PipetteSim
from .tube_association import TubeAssociationResult, associate_tubes


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy values and tuples into JSON-serializable built-ins."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class VisionEpisodeResult:
    episode_index: int
    episode_seed: int
    truth_tube_colors: dict[str, str]
    predicted_tube_colors: dict[str, str]
    association_valid: bool
    association_score: float | None
    detected_colors: tuple[str, ...]
    classification: str  # correct, safe_invalid, or wrong_valid
    failure_reason: str | None
    randomization_info: dict[str, Any]

    @property
    def correct(self) -> bool:
        return self.classification == "correct"

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class VisionEvaluationSummary:
    episodes: int
    correct: int
    safe_invalid: int
    wrong_valid: int
    all_colors_detected: int
    mean_association_score: float | None

    @property
    def correct_rate(self) -> float:
        return self.correct / self.episodes if self.episodes else 0.0

    def as_markdown(self) -> str:
        score = "n/a" if self.mean_association_score is None else f"{self.mean_association_score:.3f}"
        percent = lambda count: f"{100 * count / self.episodes:.1f}%" if self.episodes else "n/a"
        return "\n".join((
            "# Simulated RGB Vision Evaluation",
            "",
            f"Episodes: {self.episodes}",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Complete correct mapping | {self.correct} / {self.episodes} ({percent(self.correct)}) |",
            f"| Safe invalid result | {self.safe_invalid} / {self.episodes} ({percent(self.safe_invalid)}) |",
            f"| Wrong valid mapping | {self.wrong_valid} / {self.episodes} ({percent(self.wrong_valid)}) |",
            f"| All colors detected | {self.all_colors_detected} / {self.episodes} ({percent(self.all_colors_detected)}) |",
            f"| Mean valid association score | {score} |",
            "",
            "A safe invalid result is preferable to a wrong valid mapping: it lets a controller stop",
            "instead of selecting the wrong source tube.",
            "",
        ))


def summarize_results(results: list[VisionEpisodeResult]) -> VisionEvaluationSummary:
    """Aggregate episode outcomes while preserving safe rejection versus incorrect decisions."""
    valid_scores = [result.association_score for result in results if result.association_score is not None]
    return VisionEvaluationSummary(
        episodes=len(results),
        correct=sum(result.classification == "correct" for result in results),
        safe_invalid=sum(result.classification == "safe_invalid" for result in results),
        wrong_valid=sum(result.classification == "wrong_valid" for result in results),
        all_colors_detected=sum(set(COLORS).issubset(result.detected_colors) for result in results),
        mean_association_score=float(np.mean(valid_scores)) if valid_scores else None,
    )


def evaluate_episode(sim: PipetteSim, episode_index: int, episode_seed: int,
                     width: int = 640, height: int = 360, min_area_px: float = 20.0
                     ) -> tuple[VisionEpisodeResult, np.ndarray, list[BasinObservation], TubeAssociationResult]:
    """Evaluate one independently seeded randomized scene using only side-camera perception."""
    info = randomize(sim, np.random.default_rng(episode_seed))
    rgb = sim.render("side", width, height)
    observations = observe_rgb_frame(rgb, camera="side", timestamp_s=sim.data.time, min_area_px=min_area_px)
    association = associate_tubes(observations)
    truth = dict(sim.tube_colors)  # evaluation label only; never passed to perception
    predicted = association.tube_colors if association.valid else {}
    if not association.valid:
        classification, reason = "safe_invalid", association.reason
    elif predicted == truth:
        classification, reason = "correct", None
    else:
        classification, reason = "wrong_valid", "Predicted tube-to-color mapping differs from simulator truth"
    result = VisionEpisodeResult(
        episode_index=episode_index,
        episode_seed=episode_seed,
        truth_tube_colors=truth,
        predicted_tube_colors=predicted,
        association_valid=association.valid,
        association_score=association.score,
        detected_colors=tuple(sorted({observation.color for observation in observations})),
        classification=classification,
        failure_reason=reason,
        randomization_info=json_safe(info),
    )
    return result, rgb, observations, association
