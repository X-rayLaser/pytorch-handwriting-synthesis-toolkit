import math
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

    def reset(self):
        self._ma.reset()

    def prepare_arrays(self, y_hat, ground_true):
        mixture, eos = ground_true.concatenate_predictions(y_hat)
        pi, mu, sd, ro = mixture
        num_components = pi.shape[-1]

        component_indices = pi.argmax(-1).unsqueeze(1)

        mu1 = mu[:, :num_components]
        mu2 = mu[:, num_components:]

        mu1 = mu1.gather(1, component_indices)
        mu2 = mu2.gather(1, component_indices)

        predictions = torch.cat([mu1, mu2, eos], dim=1)
        actual = ground_true.concatenated()
        return predictions, actual


class MSE(Metric):
    @property
    def name(self):
        return 'MSE'

    def compute_metric(self, y_hat, ground_true):
        predictions, actual = self.prepare_arrays(y_hat, ground_true)
        return torch.nn.functional.mse_loss(predictions, actual)


class SSE(Metric):
    @property
    def name(self):
        return 'SSE'

    def compute_metric(self, y_hat, ground_true):
        predictions, actual = self.prepare_arrays(y_hat, ground_true)
        squared_errors = (predictions - actual) ** 2
        return squared_errors.sum(dim=1).mean()
