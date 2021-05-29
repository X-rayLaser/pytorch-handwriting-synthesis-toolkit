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

    def density(self, x1, x2):
        z = self.compute_z(x1, x2)
        return self.compute_density(z)

    def compute_density(self, z):
        one_minus_ro_squared = 1 - self.ro ** 2
        exp = torch.exp(-z / (2 * one_minus_ro_squared))
        denominator = 2 * math.pi * self.sd1 * self.sd2 * torch.sqrt(1 - self.ro ** 2)
        return exp / denominator

    def compute_z(self, x1, x2):
        first_term = self.normalized_square(x1, self.mu1, self.sd1)
        second_term = self.normalized_square(x2, self.mu2, self.sd2)
        substraction_term = self.substraction_term(x1, x2)
        return first_term + second_term - substraction_term

    def normalized_square(self, x, mu, sd):
        return ((x - mu) / sd) ** 2

    def substraction_term(self, x1, x2):
        return 2 * self.ro * (x1 - self.mu1) * (x2 - self.mu2) / (self.sd1 * self.sd2)


def mixture_density_loss(mixture, ground_true):
    return 0
