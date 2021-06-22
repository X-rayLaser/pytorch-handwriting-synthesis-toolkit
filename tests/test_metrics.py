import unittest

import torch

from handwriting_synthesis import metrics, utils
from handwriting_synthesis import training


class MetricTests(unittest.TestCase):
    def test_mse_name(self):
        self.assertEqual(metrics.MSE().name, 'MSE')

    def test_mse_is_zero_after_initialization(self):
        metric = metrics.MSE()
        self.assertEqual(metric.value, 0)

    def test_mse_on_a_single_example(self):
        metric = metrics.MSE()

        pi = torch.tensor([
            [[0.2, 0.7, 0.1], [0, 0.2, 0.8]],
            [[0.7, 0.2, 0.1], [0, 0.8, 0.2]]
        ])

        mu = torch.tensor([
            [[10, 5, 1, 8, 4, 2], [30, 20, 10, 3, 2, 1]],
            [[10, 5, 1, 8, 4, 2], [30, 20, 10, 3, 2, 1]]
        ])

        sd = torch.tensor([
            [[10, 5, 1, 8, 4, 2], [30, 20, 10, 3, 2, 1]],
            [[10, 5, 1, 8, 4, 2], [30, 20, 10, 3, 2, 1]]
        ])

        ro = torch.tensor([
            [[0.2, 0.2, 0.2], [0.5, 0.5, 0.5]],
            [[0.2, 0.2, 0.2], [0.5, 0.5, 0.5]]
        ])

        eos = torch.tensor([
            [[0.7], [1]],
            [[0.2], [0.5]]
        ])

        y_hat = ((pi, mu, sd, ro), eos)

        y = [
            [[5, 4, 0.2], [9, 0, 1]],
            [[10, 8, 0.7], [20, 2, 0.6]]
        ]

        padded_ground_true = utils.PaddedSequencesBatch(y)

        metric.update(y_hat, padded_ground_true)

        y_tensor = torch.tensor(y)

        expected = torch.nn.functional.mse_loss(
            torch.tensor([[(5, 4, 0.7), (10, 1, 1)], [(10, 8, 0.2), (20, 2, 0.5)]]), y_tensor
        )

        self.assertEqual(metric.value, expected)


class MovingAverageTests(unittest.TestCase):
    def test_after_initialization(self):
        ma = metrics.MovingAverage()
        self.assertEqual(ma.value, 0)

    def test_after_single_update(self):
        ma = metrics.MovingAverage()
        ma.update(25)
        self.assertEqual(ma.value, 25)

    def test_after_few_updates(self):
        ma = metrics.MovingAverage()
        ma.update(3)
        ma.update(5)
        ma.update(1)
        self.assertEqual(ma.value, 3)

    def test_reset_erases_previous_value(self):
        ma = metrics.MovingAverage()
        ma.update(3)
        ma.update(5)
        ma.reset()
        self.assertEqual(ma.value, 0)