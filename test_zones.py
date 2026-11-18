"""Tests : python -m unittest test_zones -v"""

import unittest

import numpy as np

from zones import largest_rectangle


def brute_force_area(b):
    h, w = b.shape
    best = 0
    for y0 in range(h):
        for y1 in range(y0 + 1, h + 1):
            for x0 in range(w):
                for x1 in range(x0 + 1, w + 1):
                    if b[y0:y1, x0:x1].all():
                        best = max(best, (y1 - y0) * (x1 - x0))
    return best


class TestLargestRectangle(unittest.TestCase):
    def test_hole_is_avoided(self):
        b = np.ones((5, 6), bool)
        b[2, 2] = False
        x0, y0, x1, y1 = largest_rectangle(b)
        self.assertTrue(b[y0:y1, x0:x1].all())
        self.assertEqual((x1 - x0) * (y1 - y0), 15)  # colonnes 3 à 5 sur les 5 lignes

    def test_empty_mask(self):
        self.assertIsNone(largest_rectangle(np.zeros((4, 4), bool)))

    def test_matches_brute_force_on_random_masks(self):
        rng = np.random.default_rng(0)
        for _ in range(40):
            b = rng.random((6, 7)) > 0.3
            rect = largest_rectangle(b)
            if rect is None:
                self.assertFalse(b.any())
                continue
            x0, y0, x1, y1 = rect
            self.assertTrue(b[y0:y1, x0:x1].all())  # aucun pixel vide dans le rectangle
            self.assertEqual((x1 - x0) * (y1 - y0), brute_force_area(b))  # et c'est bien le plus grand


if __name__ == "__main__":
    unittest.main()
