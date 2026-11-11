"""Integration checks for HSV perception on MuJoCo camera renders.

These tests automatically skip only when the active machine has no usable offscreen OpenGL backend.
They run normally on the Linux/EGL environment used for the project's existing scene verification.
"""
from __future__ import annotations

import unittest

import numpy as np

from pipette_sim.basin_vision import observe_rgb_frame
from pipette_sim.randomization import randomize
from pipette_sim.sim import COLORS, PipetteSim
from pipette_sim.tube_association import associate_tubes


class SimBasinVisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = PipetteSim()

    def tearDown(self) -> None:
        self.sim.close()

    def _side_observations_or_skip(self):
        try:
            image_rgb = self.sim.render("side", 640, 360)
        except Exception as error:  # backend availability is platform/display dependent
            self.skipTest(f"MuJoCo offscreen rendering unavailable: {error}")
        return observe_rgb_frame(image_rgb, camera="side", timestamp_s=self.sim.data.time, min_area_px=20.0)

    def test_nominal_side_camera_observes_all_ring_colors(self) -> None:
        observations = self._side_observations_or_skip()
        self.assertTrue(set(COLORS).issubset({observation.color for observation in observations}))
        self.assertTrue(all(observation.camera == "side" for observation in observations))
        self.assertTrue(all(0.0 <= observation.confidence <= 1.0 for observation in observations))

    def test_randomized_side_camera_observations_have_known_colors(self) -> None:
        randomize(self.sim, np.random.default_rng(7))
        observations = self._side_observations_or_skip()
        self.assertTrue(observations)
        self.assertTrue({observation.color for observation in observations}.issubset(set(COLORS)))

    def test_nominal_side_camera_associates_tube_colors(self) -> None:
        result = associate_tubes(self._side_observations_or_skip())
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.tube_colors, self.sim.tube_colors)

    def test_randomized_side_camera_associates_tube_colors(self) -> None:
        randomize(self.sim, np.random.default_rng(7))
        result = associate_tubes(self._side_observations_or_skip())
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.tube_colors, self.sim.tube_colors)
