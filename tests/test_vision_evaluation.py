"""Fast, renderer-free tests for simulated vision-evaluation metrics."""
from __future__ import annotations

import unittest

import numpy as np

from pipette_sim.vision_evaluation import VisionEpisodeResult, json_safe, summarize_results


def result(classification: str, score: float | None, colors=("blue", "green", "red")) -> VisionEpisodeResult:
    return VisionEpisodeResult(
        episode_index=0, episode_seed=0,
        truth_tube_colors={"tube_1": "red"}, predicted_tube_colors={"tube_1": "red"},
        association_valid=classification != "safe_invalid", association_score=score,
        detected_colors=colors, classification=classification,
        failure_reason=None, randomization_info={},
    )


class VisionEvaluationTests(unittest.TestCase):
    def test_summary_distinguishes_safe_rejections_from_wrong_valid_predictions(self) -> None:
        summary = summarize_results([
            result("correct", 0.9), result("safe_invalid", None, ("red",)), result("wrong_valid", 0.7),
        ])

        self.assertEqual((summary.episodes, summary.correct, summary.safe_invalid, summary.wrong_valid), (3, 1, 1, 1))
        self.assertEqual(summary.all_colors_detected, 2)
        self.assertAlmostEqual(summary.mean_association_score, 0.8)
        self.assertIn("Wrong valid mapping | 1 / 3", summary.as_markdown())

    def test_json_safe_converts_numpy_values(self) -> None:
        converted = json_safe({"values": np.array([1, 2]), "scalar": np.float32(0.5), "nested": (np.int64(3),)})
        self.assertEqual(converted, {"values": [1, 2], "scalar": 0.5, "nested": [3]})
