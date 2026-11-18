"""Tests : python -m unittest test_capture -v"""

import unittest
from types import SimpleNamespace as P

from capture import CHEEK_A, CHEEK_B, CHIN, FOREHEAD, NOSE, framing_problem, measure, pose_ok


def fake_face(nose_x=0.5, nose_y=0.55):
    """Visage factice : joues en x=0.3/0.7, front en y=0.2, menton en y=0.8."""
    lm = [P(x=0.5, y=0.5) for _ in range(478)]
    lm[CHEEK_A], lm[CHEEK_B] = P(x=0.3, y=0.5), P(x=0.7, y=0.5)
    lm[FOREHEAD], lm[CHIN] = P(x=0.5, y=0.2), P(x=0.5, y=0.8)
    lm[NOSE] = P(x=nose_x, y=nose_y)
    return lm


class TestCapture(unittest.TestCase):
    def test_measure_front(self):
        yaw, pitch, box = measure(fake_face())
        self.assertAlmostEqual(yaw, 0.5)
        self.assertAlmostEqual(pitch, (0.55 - 0.2) / 0.6)
        self.assertEqual(box, (0.3, 0.2, 0.7, 0.8))

    def test_poses(self):
        base = (0.5, 0.58)
        self.assertTrue(pose_ok("face", 0.52, 0.58, None))
        self.assertFalse(pose_ok("face", 0.75, 0.58, None))
        self.assertTrue(pose_ok("menton_leve", 0.5, 0.45, base))
        self.assertTrue(pose_ok("tete_baissee", 0.5, 0.70, base))
        self.assertTrue(pose_ok("gauche", 0.30, 0.58, base))
        self.assertTrue(pose_ok("droite", 0.70, 0.58, base))
        self.assertFalse(pose_ok("droite", 0.30, 0.58, base))

    def test_framing(self):
        self.assertIsNone(framing_problem((0.3, 0.2, 0.7, 0.8)))
        self.assertEqual(framing_problem((0.0, 0.2, 0.3, 0.8)), "Centrez votre visage")
        self.assertEqual(framing_problem((0.45, 0.45, 0.55, 0.55)), "Rapprochez-vous")
        self.assertEqual(framing_problem((0.1, 0.05, 0.9, 0.95)), "Eloignez-vous")


if __name__ == "__main__":
    unittest.main()
