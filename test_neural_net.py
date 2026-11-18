"""Tests : python -m unittest test_neural_net -v"""

import math
import unittest

from neural_net import MLP, softmax

XOR = [([0.0, 0.0], 0), ([0.0, 1.0], 1), ([1.0, 0.0], 1), ([1.0, 1.0], 0)]


class TestMLP(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        self.assertAlmostEqual(sum(softmax([1.0, 2.0, 3.0])), 1.0)

    def test_output_is_probability_distribution(self):
        p = MLP([2, 4, 3], seed=1).predict_proba([0.3, -0.7])
        self.assertEqual(len(p), 3)
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in p))

    def test_train_step_reduces_loss(self):
        net = MLP([2, 4, 2], seed=3)
        first = net.train_step([1.0, 0.0], 1, lr=0.1)
        for _ in range(50):
            last = net.train_step([1.0, 0.0], 1, lr=0.1)
        self.assertLess(last, first)
        self.assertTrue(math.isfinite(last))

    def test_learns_xor(self):
        net = MLP([2, 4, 2], seed=0).fit(XOR * 50, epochs=20, lr=0.2, seed=0)
        self.assertEqual(net.accuracy(XOR), 1.0)


if __name__ == "__main__":
    unittest.main()
