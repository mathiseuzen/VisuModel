"""Tests : python -m unittest test_isolate_skin -v"""

import unittest

import numpy as np

from isolate_skin import EXCLUDED, OVAL, to_rgba


class TestIsolateSkin(unittest.TestCase):
    def test_to_rgba_crops_and_clears_outside_mask(self):
        img = np.full((10, 10, 3), 200, np.uint8)
        mask = np.zeros((10, 10), np.uint8)
        mask[2:5, 3:7] = 255
        out = to_rgba(img, mask)
        self.assertEqual(out.shape, (3, 4, 4))  # recadré sur le masque
        self.assertTrue((out[..., 3] == 255).all())
        self.assertTrue((out[..., :3] == 200).all())

    def test_nothing_left_outside_skin(self):
        img = np.full((6, 6, 3), 99, np.uint8)
        mask = np.zeros((6, 6), np.uint8)
        mask[1, 1] = mask[4, 4] = 255
        out = to_rgba(img, mask)
        self.assertEqual(out.shape, (4, 4, 4))
        self.assertTrue((out[1, 0] == 0).all())  # pixel hors masque : transparent et noir

    def test_landmark_groups_found(self):
        self.assertGreater(len(OVAL), 20)
        self.assertEqual(len(EXCLUDED), 5)  # 2 yeux, 2 sourcils, lèvres


if __name__ == "__main__":
    unittest.main()
