"""Synthetic-image tests for the HSV basin detector (no webcam required)."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from pipette_sim.basin_vision import (annotate_observations, detected_status, detect_basins,
                                      observe_basins, observe_rgb_frame, observed_status)


class BasinVisionTests(unittest.TestCase):
    def test_detects_each_primary_color_and_ignores_tiny_region(self) -> None:
        image = np.zeros((240, 360, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 40), (100, 160), (0, 0, 255), -1)    # red BGR
        cv2.rectangle(image, (130, 40), (210, 160), (0, 255, 0), -1)   # green BGR
        cv2.rectangle(image, (240, 40), (320, 160), (255, 0, 0), -1)   # blue BGR
        cv2.circle(image, (10, 10), 3, (0, 0, 255), -1)               # below threshold

        detections = detect_basins(image, min_area_px=500)

        self.assertEqual([d.color for d in detections], ["red", "green", "blue"])
        self.assertEqual(detected_status(detections), "RED, GREEN, BLUE")
        self.assertTrue(all(d.area_px >= 500 for d in detections))

    def test_detects_red_at_hsv_wrap_boundary(self) -> None:
        hsv = np.full((100, 100, 3), (179, 220, 220), dtype=np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        detections = detect_basins(image, min_area_px=500)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].color, "red")

    def test_reports_none_for_a_neutral_image(self) -> None:
        image = np.full((100, 100, 3), 160, dtype=np.uint8)
        detections = detect_basins(image)
        self.assertEqual(detections, [])
        self.assertEqual(detected_status(detections), "NONE")

    def test_observations_preserve_detection_data_and_add_provenance(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (120, 100), (0, 0, 255), -1)

        observations = observe_basins(image, camera="side", timestamp_s=12.5, min_area_px=500)

        self.assertEqual(len(observations), 1)
        observation = observations[0]
        self.assertEqual(observation.color, "red")
        self.assertEqual(observation.camera, "side")
        self.assertEqual(observation.timestamp_s, 12.5)
        self.assertEqual(observation.image_center_px, (70, 60))
        self.assertGreaterEqual(observation.confidence, 0.0)
        self.assertLessEqual(observation.confidence, 1.0)
        self.assertIsNone(observation.tube_name)
        self.assertIsNone(observation.position_base_m)
        self.assertEqual(observed_status(observations), "RED x1")
        self.assertEqual(annotate_observations(image, observations).shape, image.shape)

    def test_rejects_empty_camera_name(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            observe_basins(image, camera="  ")

    def test_rgb_adapter_preserves_red_in_open_cv_pipeline(self) -> None:
        image_rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        image_rgb[20:101, 20:121] = (255, 0, 0)  # red in MuJoCo/Pillow-style RGB order

        observations = observe_rgb_frame(image_rgb, camera="side", timestamp_s=1.0, min_area_px=500)

        self.assertEqual([observation.color for observation in observations], ["red"])


if __name__ == "__main__":
    unittest.main()
