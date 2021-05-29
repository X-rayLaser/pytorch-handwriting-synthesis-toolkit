import unittest

import torch

import training


class MetricTests(unittest.TestCase):
    def test_mse_name(self):
        self.assertEqual(training.MSE().name, 'MSE')

    def test_mse_is_zero_after_initialization(self):
        metric = training.MSE()
        self.assertEqual(metric.value, 0)

    def test_mse_on_a_single_example(self):
        metric = training.MSE()

        y_hat = torch.tensor([[2., 3.]])
        ground_true = torch.tensor([[5., 7.]])
        metric.update(y_hat, ground_true)
        # 12.5 = (5 - 2)**2 + (7 - 3) ** 2
        self.assertEqual(metric.value, 12.5)

    def test_mse_on_a_batch(self):
        metric = training.MSE()

        y_hat = torch.tensor([[2., 3.], [1., 1.]])
        ground_true = torch.tensor([[5., 7.], [2., 2.]])
        metric.update(y_hat, ground_true)
        self.assertEqual(27 / 4, metric.value)


class MovingAverageTests(unittest.TestCase):
    def test_after_initialization(self):
        ma = training.MovingAverage()
        self.assertEqual(ma.value, 0)

    def test_after_single_update(self):
        ma = training.MovingAverage()
        ma.update(25)
        self.assertEqual(ma.value, 25)

    def test_after_few_updates(self):
        ma = training.MovingAverage()
        ma.update(3)
        ma.update(5)
        ma.update(1)
        self.assertEqual(ma.value, 3)

    def test_reset_erases_previous_value(self):
        ma = training.MovingAverage()
        ma.update(3)
        ma.update(5)
        ma.reset()
        self.assertEqual(ma.value, 0)