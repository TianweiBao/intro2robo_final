"""Unit tests for image-space association of colored observations to tube slots."""
from __future__ import annotations

import unittest

from pipette_sim.basin_vision import BasinObservation
from pipette_sim.tube_association import associate_tubes


def observation(color: str, center: tuple[int, int], confidence: float = 0.9) -> BasinObservation:
    x, y = center
    return BasinObservation(color=color, camera="side", timestamp_s=1.0, image_center_px=center,
                            bbox_px=(x - 5, y - 5, 10, 10), area_px=100.0, image_size_px=(640, 360),
                            confidence=confidence)


class TubeAssociationTests(unittest.TestCase):
    def test_assigns_aligned_row_left_to_right_despite_shuffled_colors(self) -> None:
        result = associate_tubes([
            observation("blue", (180, 101)), observation("green", (100, 100)), observation("red", (140, 102)),
        ])

        self.assertTrue(result.valid)
        self.assertEqual(result.tube_colors, {"tube_1": "green", "tube_2": "red", "tube_3": "blue"})

    def test_prefers_aligned_ring_row_over_other_same_color_regions(self) -> None:
        result = associate_tubes([
            observation("green", (100, 200), 0.90), observation("green", (100, 100), 0.85),
            observation("red", (140, 101), 0.85), observation("red", (140, 160), 0.98),
            observation("blue", (180, 99), 0.85), observation("blue", (180, 220), 0.98),
        ])

        self.assertTrue(result.valid)
        self.assertEqual([item.image_center_px for item in result.associations], [(100, 100), (140, 101), (180, 99)])

    def test_returns_invalid_result_when_a_color_is_missing(self) -> None:
        result = associate_tubes([observation("red", (100, 100)), observation("blue", (180, 100))])
        self.assertFalse(result.valid)
        self.assertIn("green", result.reason)

    def test_returns_invalid_result_when_candidates_do_not_form_row(self) -> None:
        result = associate_tubes([
            observation("green", (100, 100)), observation("red", (140, 200)), observation("blue", (180, 300)),
        ])
        self.assertFalse(result.valid)
        self.assertIn("aligned rack row", result.reason)

    def test_rejects_a_scene_spanning_false_color_region(self) -> None:
        giant_blue = BasinObservation(color="blue", camera="side", timestamp_s=1.0, image_center_px=(320, 180),
                                      bbox_px=(0, 0, 640, 360), area_px=200000.0, image_size_px=(640, 360),
                                      confidence=0.99)
        result = associate_tubes([
            observation("green", (100, 100)), observation("red", (140, 100)), giant_blue,
        ])
        self.assertFalse(result.valid)
        self.assertIn("blue", result.reason)
