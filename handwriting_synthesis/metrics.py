import torch.nn


class MovingAverage:
    def __init__(self):
        self._value = 0
        self._iterations = 0

    @property
    def value(self):
        if self._iterations == 0:
            return 0
        return self._value / self._iterations

    def update(self, v):
        self._value += v
        self._iterations += 1

    def reset(self):
        self._value = 0
        self._iterations = 0


class Metric:
    def __init__(self):
        self._ma = MovingAverage()

    @property
    def name(self):
        return ''

    @property
    def value(self):
        return self._ma.value

    def update(self, y_hat, ground_true):
        metric = self.compute_metric(y_hat, ground_true)
        self._ma.update(metric)

    def compute_metric(self, y_hat, ground_true):
        raise NotImplemented


class MSE(Metric):
    @property
    def name(self):
        return 'MSE'

    def compute_metric(self, y_hat, ground_true):
        return torch.nn.functional.mse_loss(y_hat, ground_true)