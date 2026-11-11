"""Associate colored basin observations with the known left-to-right tube rack slots.

This module deliberately operates only on :class:`BasinObservation` image measurements. It does
not import MuJoCo or use privileged simulator state, so a future physical-camera or RL pipeline can
use the same association logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .basin_vision import BasinObservation, COLOR_ORDER


@dataclass(frozen=True)
class RackAssociationConfig:
    """Image-space constraints for the one-row, three-tube rack."""

    tube_names: tuple[str, str, str] = ("tube_1", "tube_2", "tube_3")
    colors: tuple[str, str, str] = COLOR_ORDER
    max_row_spread_px: float = 30.0
    min_horizontal_spacing_px: float = 8.0
    min_confidence: float = 0.50
    max_bbox_area_fraction: float = 0.25


@dataclass(frozen=True)
class TubeAssociation:
    """One selected colored ring assigned to a known tube slot."""

    tube_name: str
    color: str
    image_center_px: tuple[int, int]
    confidence: float
    observation: BasinObservation


@dataclass(frozen=True)
class TubeAssociationResult:
    """Association result, including a safe reason when no reliable row can be selected."""

    associations: tuple[TubeAssociation, ...]
    valid: bool
    reason: str | None = None
    score: float | None = None

    @property
    def tube_colors(self) -> dict[str, str]:
        return {association.tube_name: association.color for association in self.associations}


def _invalid(reason: str) -> TubeAssociationResult:
    return TubeAssociationResult(associations=(), valid=False, reason=reason)


def associate_tubes(observations: list[BasinObservation],
                    config: RackAssociationConfig = RackAssociationConfig()) -> TubeAssociationResult:
    """Choose one aligned observation per color and map left-to-right centers to rack slots.

    Multiple contours of the same color are expected because the simulator can show both colored
    liquid and the matching rim ring. Candidate triples are scored by confidence, common image row,
    and regular horizontal spacing. Ambiguous or incomplete evidence returns ``valid=False``.
    """
    if len(config.tube_names) != len(config.colors) or len(config.colors) != 3:
        raise ValueError("This rack association requires exactly three tube names and three colors")

    def plausible_target(observation: BasinObservation) -> bool:
        _, _, width, height = observation.bbox_px
        image_width, image_height = observation.image_size_px
        bbox_fraction = width * height / (image_width * image_height)
        return observation.confidence >= config.min_confidence and bbox_fraction <= config.max_bbox_area_fraction

    candidates_by_color = {
        color: [observation for observation in observations if observation.color == color and plausible_target(observation)]
        for color in config.colors
    }
    missing = [color for color, candidates in candidates_by_color.items() if not candidates]
    if missing:
        return _invalid(f"Missing plausible target observation for: {', '.join(missing)}")

    best: tuple[float, tuple[BasinObservation, ...]] | None = None
    for candidate_row in product(*(candidates_by_color[color] for color in config.colors)):
        centers = [candidate.image_center_px for candidate in candidate_row]
        xs, ys = (sorted(center[0] for center in centers), [center[1] for center in centers])
        row_spread = max(ys) - min(ys)
        if row_spread > config.max_row_spread_px:
            continue
        gap_left, gap_right = xs[1] - xs[0], xs[2] - xs[1]
        if min(gap_left, gap_right) < config.min_horizontal_spacing_px:
            continue

        confidence_score = sum(candidate.confidence for candidate in candidate_row) / len(candidate_row)
        alignment_score = 1.0 - row_spread / config.max_row_spread_px
        spacing_score = 1.0 - abs(gap_left - gap_right) / max(gap_left, gap_right)
        score = 0.55 * confidence_score + 0.30 * alignment_score + 0.15 * spacing_score
        if best is None or score > best[0]:
            best = (score, candidate_row)

    if best is None:
        return _invalid("No red/green/blue candidates form a valid aligned rack row")

    score, selected = best
    selected_by_x = sorted(selected, key=lambda observation: observation.image_center_px[0])
    associations = tuple(
        TubeAssociation(tube_name=tube, color=observation.color,
                        image_center_px=observation.image_center_px,
                        confidence=observation.confidence, observation=observation)
        for tube, observation in zip(config.tube_names, selected_by_x)
    )
    return TubeAssociationResult(associations=associations, valid=True, score=score)
