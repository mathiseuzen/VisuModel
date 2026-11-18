"""Tests : python -m unittest test_train_patch_cnn -v"""

import unittest

import numpy as np
import torch

from train_patch_cnn import PatchCNN, confusion, scores


class TestTraining(unittest.TestCase):
    def test_output_shape_and_other_sizes(self):
        net = PatchCNN(7)
        self.assertEqual(net(torch.zeros(2, 3, 64, 64)).shape, (2, 7))
        self.assertEqual(net(torch.zeros(2, 3, 48, 48)).shape, (2, 7))

    def test_confusion_and_scores(self):
        # vrai : 0 0 0 1 1 2 ; prédit : 0 0 1 1 1 0
        m = confusion([0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 2], 3)
        np.testing.assert_array_equal(m, [[2, 1, 0], [0, 2, 0], [1, 0, 0]])
        precision, recall, f1 = scores(m)
        np.testing.assert_allclose(precision, [2 / 3, 2 / 3, 0])
        np.testing.assert_allclose(recall, [2 / 3, 1.0, 0])
        self.assertAlmostEqual(f1[1], 0.8)
        self.assertEqual(f1[2], 0)  # classe jamais bien prédite : 0, pas d'erreur de division

    def test_overfits_a_tiny_batch(self):
        torch.manual_seed(0)
        net, loss_fn = PatchCNN(3), torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)
        x, y = torch.randn(12, 3, 64, 64), torch.randint(0, 3, (12,))
        first = loss_fn(net(x), y).item()
        for _ in range(40):
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), first / 2)


if __name__ == "__main__":
    unittest.main()
