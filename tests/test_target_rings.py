"""Simulator-level checks for the colored tube-opening target rings."""
from __future__ import annotations

import unittest

import numpy as np

from pipette_sim import specs as S
from pipette_sim.sim import TUBES, PipetteSim


class TargetRingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = PipetteSim()

    def tearDown(self) -> None:
        self.sim.close()

    def test_default_rings_match_default_tube_liquids(self) -> None:
        for tube in TUBES:
            ring_rgb = self.sim.model.geom(f"{tube}_target_ring").rgba[:3]
            self.assertTrue(np.allclose(ring_rgb, S.LIQUID_COLORS[self.sim.tube_colors[tube]]))

    def test_reset_updates_rings_for_shuffled_tube_colors(self) -> None:
        colors = ("blue", "green", "red")
        self.sim.reset(tube_colors=colors)
        for tube, color in zip(TUBES, colors):
            ring_rgb = self.sim.model.geom(f"{tube}_target_ring").rgba[:3]
            self.assertTrue(np.allclose(ring_rgb, S.LIQUID_COLORS[color]))


if __name__ == "__main__":
    unittest.main()
