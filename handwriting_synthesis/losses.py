import math
import torch


class BiVariateGaussian:
    @classmethod
    def from_scalars(cls, mu, sd, ro):
        mu1, mu2 = mu
        sd1, sd2 = sd

        mu = (cls.to_tensor(mu1), cls.to_tensor(mu2))
        sd = (cls.to_tensor(sd1), cls.to_tensor(sd2))
        ro = cls.to_tensor(ro)
        return BiVariateGaussian(mu, sd, ro)

    @classmethod
    def to_tensor(cls, scalar):
        return torch.tensor(scalar, dtype=torch.float64)

    def __init__(self, mu, sd, ro):
        """

        :param mu: Tuple[Tensor, Tensor]
        :param sd: Tuple[Tensor, Tensor]
        :param ro: Tensor
        """
        self.mu1, self.mu2 = mu
        self.sd1, self.sd2 = sd
        self.ro = ro

        self.epsilon = 10**(-8)

    def density(self, x1, x2):
        z = self.compute_z(x1, x2)
        return self.compute_density(z)

    def compute_density(self, z):
        eps = self.epsilon
        one_minus_ro_squared = 1 - self.ro ** 2
        exp = torch.exp(-z / (2 * one_minus_ro_squared + eps))
        denominator = 2 * math.pi * self.sd1 * self.sd2 * torch.sqrt(1 - self.ro ** 2) + eps
        return exp / denominator

    def compute_z(self, x1, x2):
        first_term = self.normalized_square(x1, self.mu1, self.sd1)
        second_term = self.normalized_square(x2, self.mu2, self.sd2)
        substraction_term = self.substraction_term(x1, x2)
        return first_term + second_term - substraction_term

    def normalized_square(self, x, mu, sd):
        sd = sd + self.epsilon
        return ((x - mu) / sd) ** 2

    def substraction_term(self, x1, x2):
        return 2 * self.ro * (x1 - self.mu1) * (x2 - self.mu2) / (self.sd1 * self.sd2 + self.epsilon)


class Mixture:
    def __init__(self, pi, mu, sd, ro):
        """

        :param pi: 2D Tensor of shape (num_steps, num_components)
        :param mu: 2D Tensor of shape (num_steps, num_components * 2)
        :param sd: 2D Tensor of shape (num_steps, num_components * 2)
        :param ro: 2D Tensor of shape (num_steps, num_components)
        """
        self.pi = pi
        self.mu = mu
        self.sd = sd
        self.ro = ro

    @property
    def num_components(self):
        return self.pi.shape[-1]

    @property
    def mu1(self):
        return self.mu[:, :self.num_components]

    @property
    def mu2(self):
        return self.mu[:, self.num_components:]

    @property
    def sd1(self):
        return self.sd[:, :self.num_components]

    @property
    def sd2(self):
        return self.sd[:, self.num_components:]

    def log_density(self, x1, x2):
        """

        :param x1: 1D Tensor
        :param x2: 1D Tensor
        :return: scalar
        """
        eps = 10**(-16)
        x1 = self._prepare_x(x1)
        x2 = self._prepare_x(x2)
        mu = (self.mu1, self.mu2)
        sd = (self.sd1, self.sd2)
        gaussian = BiVariateGaussian(mu, sd, self.ro)
        densities = gaussian.density(x1, x2) + eps
        mixture_densities = (densities * self.pi).sum(dim=1)
        return torch.log(mixture_densities + eps).sum(dim=0)

    def _prepare_x(self, x):
        return x.unsqueeze(1).repeat(1, self.num_components)


def nll_loss(mixture, eos_hat, ground_true):
    """
    Compute a negative log likelihood loss for a batch of observed sequences.

    :param mixture: A tuple of (pi, mu, sd, ro) tensors
    :param eos_hat: End-Of-Stroke predictions, torch.Tensor of shape (batch_size, sequence_size, 1)
    :param ground_true: instance of utils.PaddedSequencesBatch
    :return: a scalar storing the value for a negative log likelihood
    """
    y = ground_true.concatenated()
    pi, mu, sd, ro = mixture
    batch_size = len(pi)

    num_components = pi.shape[-1]
    mu1 = mu[:, :, :num_components]
    mu2 = mu[:, :, num_components:]

    sd1 = sd[:, :, :num_components]
    sd2 = sd[:, :, num_components:]

    pi = ground_true.concatenate_batch(pi)
    mu1 = ground_true.concatenate_batch(mu1)
    mu2 = ground_true.concatenate_batch(mu2)
    mu = torch.cat([mu1, mu2], dim=1)

    sd1 = ground_true.concatenate_batch(sd1)
    sd2 = ground_true.concatenate_batch(sd2)
    sd = torch.cat([sd1, sd2], dim=1)

    ro = ground_true.concatenate_batch(ro)

    y1 = y[:, 0]
    y2 = y[:, 1]
    eos = y[:, 2]

    eos_hat = ground_true.concatenate_batch(eos_hat)
    eos_hat = eos_hat[:, 0]

    gaussian_mixture = Mixture(pi, mu, sd, ro)
    density = gaussian_mixture.log_density(y1, y2)

    eps = 10 ** (-8)
    binary_log_likelihood = (eos * torch.log(eos_hat + eps) + (1 - eos) * torch.log(1 - eos_hat + eps)).sum()

    return -(density + binary_log_likelihood) / batch_size
