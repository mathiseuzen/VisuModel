"""Tests : python -m unittest test_prepare_patches -v"""

import random
import unittest

import numpy as np

from prepare_patches import MAX_OVERLAP, MIN_SIDE, PATCH, SKIN_SHARE, crop_square, overlap_ratio, sample_healthy


class TestPatches(unittest.TestCase):
    def test_crop_square_shape_and_borders(self):
        img = np.arange(200 * 300 * 3, dtype=np.uint8).reshape(200, 300, 3)
        self.assertEqual(crop_square(img, 150, 100, 50).shape, (PATCH, PATCH, 3))
        self.assertEqual(crop_square(img, 0, 0, 80).shape, (PATCH, PATCH, 3))        # coin : le carré reste dans l'image
        self.assertEqual(crop_square(img, 299, 199, 9999).shape, (PATCH, PATCH, 3))  # côté trop grand : borné à l'image
        self.assertEqual(crop_square(img, 10, 10, 2).shape, (PATCH, PATCH, 3))       # côté trop petit : relevé à MIN_SIDE

    def test_overlap_ratio(self):
        self.assertEqual(overlap_ratio((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(overlap_ratio((0, 0, 10, 10), (5, 0, 15, 10)), 0.5)
        self.assertEqual(overlap_ratio((10, 10, 20, 20), (0, 0, 100, 100)), 1.0)  # petit carré dans une grande boîte
        self.assertEqual(overlap_ratio((0, 0, 100, 100), (10, 10, 20, 20)), 1.0)  # petite boîte dans un grand carré

    def test_healthy_samples_avoid_boxes_and_non_skin(self):
        skin = np.ones((300, 300), bool)
        skin[:, 200:] = False                  # moitié droite : pas de peau
        boxes = [(0, 0, 100, 300)]             # bande gauche occupée par une boîte
        found = sample_healthy(skin, boxes, [MIN_SIDE], 20, random.Random(0))
        self.assertGreater(len(found), 0)
        for x0, y0, side in found:
            self.assertLessEqual(overlap_ratio((x0, y0, x0 + side, y0 + side), boxes[0]), MAX_OVERLAP)  # pas sur la boîte
            self.assertLessEqual(x0 + side, 200 + side * (1 - SKIN_SHARE))  # pas (ou presque pas) hors peau
        self.assertEqual(sample_healthy(np.zeros((300, 300), bool), [], [MIN_SIDE], 5, random.Random(0)), [])


if __name__ == "__main__":
    unittest.main()
