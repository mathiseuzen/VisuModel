"""Tests : python -m unittest test_detect -v"""

import unittest

import numpy as np
import torch

from detect import classify, iou, nms, skin_windows
from train_patch_cnn import PatchCNN


class TestDetect(unittest.TestCase):
    def test_windows_are_fully_on_skin(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[20:60, 30:90] = 255
        found = skin_windows(mask, 20, 10)
        self.assertGreater(len(found), 0)
        for x0, y0 in found:
            self.assertTrue((mask[y0:y0 + 20, x0:x0 + 20] > 0).all())

    def test_no_window_when_skin_too_small_or_absent(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[10:20, 10:20] = 255  # 10x10 : plus petit que la fenêtre
        self.assertEqual(skin_windows(mask, 20, 5), [])
        self.assertEqual(skin_windows(np.zeros((50, 50), np.uint8), 20, 5), [])

    def test_hole_is_avoided(self):
        mask = np.full((60, 60), 255, np.uint8)
        mask[28:32, 28:32] = 0  # un trou (comme un sourcil)
        for x0, y0 in skin_windows(mask, 20, 5):
            self.assertTrue((mask[y0:y0 + 20, x0:x0 + 20] > 0).all())

    def test_iou_and_nms(self):
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
        self.assertEqual(sorted(nms(boxes, np.array([0.9, 0.95, 0.8]))), [1, 2])  # la moins confiante des 2 voisines part

    def test_classify_shape_and_probabilities(self):
        crops = [np.zeros((64, 64, 3), np.uint8) for _ in range(5)]
        probs = classify(PatchCNN(7).eval(), crops)
        self.assertEqual(probs.shape, (5, 7))
        np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
