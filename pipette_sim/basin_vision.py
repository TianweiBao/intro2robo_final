"""HSV-based red, green, and blue basin-marker detection.

The functions in this module accept ordinary OpenCV BGR images, so the same detector can be used
with a laptop webcam today and with a robot or MuJoCo camera stream later.  It intentionally only
finds colored regions; mapping a region to a robot-frame tube location belongs in a later calibrated
perception/control layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVRange:
    """Inclusive OpenCV HSV range (H: 0--179; S/V: 0--255)."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass(frozen=True)
class BasinDetection:
    """Low-level connected colored region detected in an image."""

    color: str
    bbox: tuple[int, int, int, int]  # x, y, width, height in pixels
    centroid: tuple[int, int]
    area_px: float
    solidity: float
    touches_image_edge: bool

    @property
    def label(self) -> str:
        return self.color.upper()


@dataclass(frozen=True)
class BasinObservation:
    """Camera-aware basin target available to later association and motion-control layers.

    Image positions use OpenCV's top-left origin and ``(x, y)`` order. ``tube_name`` and
    ``position_base_m`` intentionally remain unset until future tube-association and calibration
    stages establish those values.
    """

    color: str
    camera: str
    timestamp_s: float
    image_center_px: tuple[int, int]
    bbox_px: tuple[int, int, int, int]
    area_px: float
    image_size_px: tuple[int, int]
    confidence: float
    tube_name: str | None = None
    position_base_m: np.ndarray | None = None


# Red spans the HSV hue discontinuity, hence its two intervals. These defaults deliberately require
# reasonably saturated, non-dark colors so neutral lab equipment is not selected.
DEFAULT_HSV_RANGES: dict[str, tuple[HSVRange, ...]] = {
    "red": (HSVRange((0, 100, 70), (10, 255, 255)), HSVRange((170, 100, 70), (179, 255, 255))),
    "green": (HSVRange((40, 80, 60), (85, 255, 255)),),
    "blue": (HSVRange((95, 100, 60), (135, 255, 255)),),
}

DRAW_BGR = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0)}
COLOR_ORDER = ("red", "green", "blue")
TUBE_CALLOUT_BGR = (0, 150, 190)  # dark yellow in OpenCV BGR order


def color_mask(hsv_frame: np.ndarray, color: str,
               hsv_ranges: dict[str, tuple[HSVRange, ...]] = DEFAULT_HSV_RANGES) -> np.ndarray:
    """Return a cleaned binary mask for one named color in an OpenCV HSV image."""
    if color not in hsv_ranges:
        raise ValueError(f"Unsupported color {color!r}; expected one of {tuple(hsv_ranges)}")
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for interval in hsv_ranges[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, interval.lower, interval.upper))
    # A 3x3 kernel removes isolated pixels without erasing the deliberately narrow tube-rim
    # targets in simulator and wrist-camera views.
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def detect_basins(frame_bgr: np.ndarray, min_area_px: float = 500.0,
                  hsv_ranges: dict[str, tuple[HSVRange, ...]] = DEFAULT_HSV_RANGES) -> list[BasinDetection]:
    """Detect colored regions larger than ``min_area_px`` in a BGR image.

    Multiple regions of the same color are retained. Results are ordered RED, GREEN, BLUE and then
    by descending area, making status displays and downstream selection deterministic.
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR image")
    if min_area_px < 0:
        raise ValueError("min_area_px must be non-negative")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    detections: list[BasinDetection] = []
    for color in COLOR_ORDER:
        if color not in hsv_ranges:
            continue
        contours, _ = cv2.findContours(color_mask(hsv, color, hsv_ranges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[BasinDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            cx = int(moments["m10"] / moments["m00"]) if moments["m00"] else x + w // 2
            cy = int(moments["m01"] / moments["m00"]) if moments["m00"] else y + h // 2
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / hull_area if hull_area > 0 else 0.0
            image_h, image_w = frame_bgr.shape[:2]
            touches_edge = x <= 0 or y <= 0 or x + w >= image_w or y + h >= image_h
            candidates.append(BasinDetection(color, (x, y, w, h), (cx, cy), area, solidity, touches_edge))
        detections.extend(sorted(candidates, key=lambda detection: detection.area_px, reverse=True))
    return detections


def detected_status(detections: list[BasinDetection]) -> str:
    """Return the concise text for a UI status line."""
    colors = [color.upper() for color in COLOR_ORDER if any(d.color == color for d in detections)]
    return ", ".join(colors) if colors else "NONE"


def detection_confidence(detection: BasinDetection, min_area_px: float) -> float:
    """Return an explainable 0--1 quality score for a contour detection.

    The score rewards a compact contour (solidity), a region at least a few times larger than the
    area cutoff, and a target that is fully inside the camera image. It is a quality heuristic, not
    a learned probability of correctness.
    """
    if min_area_px <= 0:
        raise ValueError("min_area_px must be positive when computing confidence")
    area_score = min(1.0, detection.area_px / (3.0 * min_area_px))
    shape_score = float(np.clip(detection.solidity, 0.0, 1.0))
    edge_score = 0.65 if detection.touches_image_edge else 1.0
    return float(np.clip(0.40 * area_score + 0.45 * shape_score + 0.15 * edge_score, 0.0, 1.0))


def observe_basins(frame_bgr: np.ndarray, camera: str, timestamp_s: float | None = None,
                   min_area_px: float = 500.0,
                   hsv_ranges: dict[str, tuple[HSVRange, ...]] = DEFAULT_HSV_RANGES) -> list[BasinObservation]:
    """Convert HSV detections from one frame into camera-aware basin observations."""
    if not camera.strip():
        raise ValueError("camera must be a non-empty name")
    timestamp_s = time.monotonic() if timestamp_s is None else float(timestamp_s)
    detections = detect_basins(frame_bgr, min_area_px, hsv_ranges)
    return [BasinObservation(
        color=detection.color,
        camera=camera,
        timestamp_s=timestamp_s,
        image_center_px=detection.centroid,
        bbox_px=detection.bbox,
        area_px=detection.area_px,
        image_size_px=(frame_bgr.shape[1], frame_bgr.shape[0]),
        confidence=detection_confidence(detection, min_area_px),
    ) for detection in detections]


def observe_rgb_frame(frame_rgb: np.ndarray, camera: str, timestamp_s: float | None = None,
                      min_area_px: float = 500.0,
                      hsv_ranges: dict[str, tuple[HSVRange, ...]] = DEFAULT_HSV_RANGES) -> list[BasinObservation]:
    """Observe an RGB frame, including images returned by :meth:`PipetteSim.render`.

    MuJoCo returns RGB while OpenCV's segmentation routines expect BGR. Keeping this conversion in
    one reusable adapter prevents an accidental red/blue swap in demos or a future RL environment.
    """
    if frame_rgb is None or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must be an HxWx3 RGB image")
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return observe_basins(frame_bgr, camera, timestamp_s, min_area_px, hsv_ranges)


def observed_status(observations: list[BasinObservation]) -> str:
    """Return a compact UI status line grouped by detected color."""
    if not observations:
        return "NONE"
    return ", ".join(
        f"{color.upper()} x{sum(observation.color == color for observation in observations)}"
        for color in COLOR_ORDER if any(observation.color == color for observation in observations)
    )


def annotate_detections(frame_bgr: np.ndarray, detections: list[BasinDetection]) -> np.ndarray:
    """Return an annotated copy of a BGR frame with boxes, labels, and overall status."""
    annotated = frame_bgr.copy()
    for detection in detections:
        x, y, w, h = detection.bbox
        color = DRAW_BGR[detection.color]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label_y = max(22, y - 8)
        cv2.putText(annotated, detection.label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Currently detected basins: {detected_status(detections)}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(annotated, f"Currently detected basins: {detected_status(detections)}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    return annotated


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Whether two ``(left, top, right, bottom)`` rectangles overlap."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _nearest_point(rect: tuple[int, int, int, int], point: tuple[int, int]) -> tuple[int, int]:
    return (min(max(point[0], rect[0]), rect[2]), min(max(point[1], rect[1]), rect[3]))


def _draw_callout(annotated: np.ndarray, bbox: tuple[int, int, int, int], text: str,
                  color: tuple[int, int, int], font_scale: float,
                  occupied: list[tuple[int, int, int, int]], blocked: list[tuple[int, int, int, int]]) -> None:
    """Draw one non-overlapping text callout with a leader line to its source box."""
    x, y, w, h = bbox
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    label_w, label_h, pad = text_size[0] + 8, text_size[1] + baseline + 8, 4
    image_h, image_w = annotated.shape[:2]
    source_rect = (x, y, x + w, y + h)
    # Try adjacent positions first, then progressively further away. The header is reserved so a
    # moved label never overwrites the overall detection status.
    candidates = []
    for distance in (6, 22, 42, 66, 94):
        candidates.extend([
            (x, y - label_h - distance), (x, y + h + distance),
            (x + w + distance, y), (x - label_w - distance, y),
        ])
    label_rect = None
    for left, top in candidates:
        left = min(max(left, 0), image_w - label_w)
        top = min(max(top, 38), image_h - label_h)
        candidate = (left, top, left + label_w, top + label_h)
        if not any(_intersects(candidate, existing) for existing in occupied + blocked):
            label_rect = candidate
            break
    # Dense scenes can exhaust all ideal positions. Preserve readable text by only relaxing the
    # no-geometry rule; never permit overlap with another text label.
    if label_rect is None:
        for left, top in candidates:
            left = min(max(left, 0), image_w - label_w)
            top = min(max(top, 38), image_h - label_h)
            candidate = (left, top, left + label_w, top + label_h)
            if not any(_intersects(candidate, existing) for existing in occupied):
                label_rect = candidate
                break
    if label_rect is None:  # pragma: no cover - only reachable with an exceptionally tiny image
        return

    center = (x + w // 2, y + h // 2)
    line_start = _nearest_point(label_rect, center)
    cv2.arrowedLine(annotated, line_start, center, color, 1, cv2.LINE_AA, tipLength=0.18)
    cv2.rectangle(annotated, (label_rect[0], label_rect[1]), (label_rect[2], label_rect[3]), (20, 20, 20), -1)
    cv2.rectangle(annotated, (label_rect[0], label_rect[1]), (label_rect[2], label_rect[3]), color, 1)
    cv2.putText(annotated, text, (label_rect[0] + pad, label_rect[1] + pad + text_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    occupied.append(label_rect)


def annotate_observations(frame_bgr: np.ndarray, observations: list[BasinObservation],
                          tube_callouts: list[tuple[BasinObservation, str]] | None = None) -> np.ndarray:
    """Return an annotated BGR frame with non-overlapping confidence and tube callouts.

    ``tube_callouts`` accepts selected observations paired with a tube label. Those labels are
    dark yellow; all labels receive leader lines so the source box remains unambiguous when text is
    moved to avoid an overlap.
    """
    annotated = frame_bgr.copy()
    for observation in observations:
        x, y, w, h = observation.bbox_px
        color = DRAW_BGR[observation.color]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    status = f"Currently detected basins: {observed_status(observations)}"
    cv2.putText(annotated, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(annotated, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    occupied: list[tuple[int, int, int, int]] = []
    blocked = [(0, 0, annotated.shape[1], 36)] + [
        (observation.bbox_px[0] - 2, observation.bbox_px[1] - 2,
         observation.bbox_px[0] + observation.bbox_px[2] + 2,
         observation.bbox_px[1] + observation.bbox_px[3] + 2)
        for observation in observations
    ]
    # Prioritize the selected tube labels, then place compact raw color/confidence labels.
    for observation, text in tube_callouts or []:
        _draw_callout(annotated, observation.bbox_px, text, TUBE_CALLOUT_BGR, 0.42, occupied, blocked)
    for observation in observations:
        _draw_callout(annotated, observation.bbox_px, f"{observation.color.upper()} {observation.confidence:.2f}",
                      DRAW_BGR[observation.color], 0.36, occupied, blocked)
    return annotated
