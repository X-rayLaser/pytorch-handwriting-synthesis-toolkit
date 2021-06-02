import unittest
import torch
from handwriting_synthesis import losses
import math

from handwriting_synthesis import utils


class BiVariateNormalZTests(unittest.TestCase):
    def test_normalized_square_helper_method(self):
        mu = (1., 4.)
        sd = (2., 4.)
        ro = 0.

        x1 = 4.
        gaussian = losses.BiVariateGaussian(mu, sd, ro)

        res = gaussian.normalized_square(x1, mu[0], sd[0])
        expected = ((x1 - mu[0]) / sd[0]) ** 2
        self.assertAlmostEqual(res, expected, places=6)
        self.assertAlmostEqual(res, 1.5 ** 2, places=6)

    def test_normalized_square_helper_method_avoids_division_by_zero(self):
        mu = (1., 4.)
        sd = (0., 4.)
        ro = 0.

        x1 = 4.
        gaussian = losses.BiVariateGaussian(mu, sd, ro)

        res = gaussian.normalized_square(x1, mu[0], sd[0])
        expected = ((x1 - mu[0]) / (sd[0] + gaussian.epsilon)) ** 2
        self.assertAlmostEqual(res, expected, places=6)
        self.assertAlmostEqual(res, (3.0 / gaussian.epsilon) ** 2, places=6)

    def test_substraction_term_helper_method(self):
        mu = (1., 4.)
        sd = (2., 3.)
        ro = 0.5

        x1, x2 = (4., 5.)
        gaussian = losses.BiVariateGaussian(mu, sd, ro)

        res = gaussian.substraction_term(x1, x2)
        expected = 2 * ro * (x1 - mu[0]) * (x2 - mu[1]) / (sd[0] * sd[1])
        self.assertAlmostEqual(res, expected, places=6)
        self.assertAlmostEqual(res, 0.5, places=6)

    def test_substraction_term_helper_method_avoids_division_by_zero(self):
        mu = (3., 4.)
        sd = (0., 0.)
        ro = 0.5

        x1, x2 = (4., 5.)
        gaussian = losses.BiVariateGaussian(mu, sd, ro)

        res = gaussian.substraction_term(x1, x2)
        expected = 2 * ro * (x1 - mu[0]) * (x2 - mu[1]) / (sd[0] * sd[1] + gaussian.epsilon)
        self.assertAlmostEqual(res, expected, places=6)
        self.assertAlmostEqual(res, 1.0 / gaussian.epsilon, places=6)

    def test_z_formula_uses_first_term(self):
        mu = (1., 4.)
        sd = (2., 4.)
        ro = 0.

        x1 = 4.
        x2 = 4.
        gaussian = losses.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        expected = 9/4.0
        self.assertAlmostEqual(z, expected, places=6)
        self.assertAlmostEqual(z, (x1 - mu[0]) ** 2 / sd[0] ** 2, places=6)

    def test_z_formula_uses_second_term(self):
        mu = (4., 4.)
        sd = (2., 4.)
        ro = 0.

        x1, x2 = (4., 1.)
        gaussian = losses.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        expected = 9 / 16.
        self.assertAlmostEqual(z, expected, places=6)
        self.assertAlmostEqual(z, (x2 - mu[1]) ** 2 / sd[1] ** 2, places=6)

    def test_z_formula_uses_third_term(self):
        mu = (2., 4.)
        sd = (2., 3.)
        ro = 1

        x1, x2 = (4., 1.)
        gaussian = losses.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        expected = 1 + 1 - (-2)
        self.assertAlmostEqual(z, expected, places=6)
        expected = (gaussian.normalized_square(x1, mu[0], sd[0])
                    + gaussian.normalized_square(x2, mu[1], sd[1])
                    - 2 * ro * (x1 - mu[0]) * (x2 - mu[1]) / (sd[0] * sd[1]))
        self.assertAlmostEqual(z, expected, places=6)

    def test_formula_on_tensors(self):
        mu1 = torch.tensor([2., 5.])
        mu2 = torch.tensor([4., 6.])
        mu = (mu1, mu2)

        sd1 = torch.tensor([2., 4.])
        sd2 = torch.tensor([3., 3.])
        sd = (sd1, sd2)

        ro = torch.tensor([1., 0.5])

        x1 = torch.tensor([4., 6.5])
        x2 = torch.tensor([1., 3.])

        gaussian = losses.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        first_component = losses.BiVariateGaussian((2., 4.), (2., 3.), 1.)
        second_component = losses.BiVariateGaussian((5., 6.), (4., 3.), 0.5)

        res1 = first_component.compute_z(x1[0], x2[0])
        res2 = second_component.compute_z(x1[1], x2[1])

        self.assertEqual(z.shape, (2,))
        self.assertEqual(z[0], res1)
        self.assertEqual(z[1], res2)

    def test_formula_on_3d_tensors(self):
        shape = (3, 5, 10)

        mu1_value = 4.
        mu2_value = 3.
        mu1 = torch.ones(*shape) * mu1_value
        mu2 = torch.ones(*shape) * mu2_value
        mu = (mu1, mu2)

        sd1_value = 2.
        sd2_value = 3.
        sd1 = torch.ones(*shape) * sd1_value
        sd2 = torch.ones(*shape) * sd2_value
        sd = (sd1, sd2)

        ro_value = 0.4
        ro = torch.ones(*shape) * ro_value

        x1_value = -2
        x2_value = 6
        x1 = torch.ones(*shape) * x1_value
        x2 = torch.ones(*shape) * x2_value

        gaussian = losses.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        scalar_gaussian = losses.BiVariateGaussian(
            mu=(mu1_value, mu2_value), sd=(sd1_value, sd2_value), ro=ro_value
        )
        expected_z = (scalar_gaussian.normalized_square(x1_value, mu1_value, sd1_value)
                      + scalar_gaussian.normalized_square(x2_value, mu2_value, sd2_value)
                      - scalar_gaussian.substraction_term(x1_value, x2_value))

        expected_z = torch.ones(shape) * expected_z
        self.assertTupleEqual(z.shape, shape)
        self.assertTrue(torch.allclose(expected_z, z))


class BiVariateNormalScalarTests(unittest.TestCase):
    def test_compute_density_formula_accounts_for_denominator(self):
        mu = (3., 4.)
        sd = (2., 3.)
        sd1, sd2 = sd
        ro = 0.5
        z = 0
        gaussian = losses.BiVariateGaussian.from_scalars(mu, sd, ro)
        density = gaussian.compute_density(torch.tensor(z)).item()
        expected = 1.0 / (2 * math.pi * sd1 * sd2 * math.sqrt(1 - ro ** 2))
        self.assertAlmostEqual(density, expected, places=6)

    def test_compute_density_formula_correct(self):
        mu = (3., 4.)
        sd = (2., 3.)
        sd1, sd2 = sd
        ro = 0.5
        z = 3.2
        gaussian = losses.BiVariateGaussian.from_scalars(mu, sd, ro)
        density = gaussian.compute_density(gaussian.to_tensor(z))

        one_minus_ro_squared = 1 - ro ** 2
        exp = math.exp(-z / (2 * one_minus_ro_squared))
        denominator = (2 * math.pi * sd1 * sd2 * math.sqrt(1 - ro ** 2))
        expected = exp / denominator
        self.assertAlmostEqual(density.item(), expected, places=8)

    def test_density_formula_avoids_division_by_zero(self):
        mu = (3., 4.)
        sd = (2., 3.)
        sd1, sd2 = sd
        ro = 1
        z = 3.2
        gaussian = losses.BiVariateGaussian.from_scalars(mu, sd, ro)
        density = gaussian.compute_density(gaussian.to_tensor(z))

        one_minus_ro_squared = 1 - ro ** 2
        exp = math.exp(-z / (2 * one_minus_ro_squared + gaussian.epsilon))
        denominator = (2 * math.pi * sd1 * sd2 * math.sqrt(1 - ro ** 2)) + gaussian.epsilon
        expected = exp / denominator
        self.assertAlmostEqual(density.item(), expected, places=8)

    def test_density_on_scalar_inputs(self):
        mu = (3., 4.)
        sd = (2., 3.)
        ro = 0.5

        gaussian = losses.BiVariateGaussian.from_scalars(mu, sd, ro)
        x1, x2 = (-4., -2)
        x1, x2 = gaussian.to_tensor(x1), gaussian.to_tensor(x2)
        density = gaussian.density(x1, x2)
        expected = gaussian.compute_density(gaussian.compute_z(x1, x2))

        self.assertEqual(density, expected)

    def test_density_on_3d_tensor(self):
        shape = (3, 5, 10)

        mu1_value = 3.
        mu2_value = 5.
        mu1 = torch.ones(*shape) * mu1_value
        mu2 = torch.ones(*shape) * mu2_value
        mu = (mu1, mu2)

        sd1_value = 2.
        sd2_value = 3.
        sd1 = torch.ones(*shape) * sd1_value
        sd2 = torch.ones(*shape) * sd2_value
        sd = (sd1, sd2)

        ro_value = 0.4

        ro = torch.ones(*shape) * ro_value

        x1_value, x2_value = (-4., -2.)
        x1 = torch.ones(*shape) * x1_value
        x2 = torch.ones(*shape) * x2_value

        gaussian = losses.BiVariateGaussian(mu, sd, ro)
        scalar_gaussian = losses.BiVariateGaussian.from_scalars(
            mu=(mu1_value, mu2_value), sd=(sd1_value, sd2_value), ro=ro_value
        )

        density = gaussian.density(x1, x2)
        scalar_density = scalar_gaussian.density(scalar_gaussian.to_tensor(x1_value),
                                                 scalar_gaussian.to_tensor(x2_value))
        expected = torch.ones(*shape) * scalar_density.item()

        self.assertTupleEqual(density.shape, expected.shape)
        self.assertTupleEqual(density.shape, shape)
        self.assertTrue(torch.allclose(density, expected))

    def test_very_special_case(self):
        mu = (3., 4.)
        sd = (1., 1.)
        ro = 0.

        x1 = 3.
        x2 = 4.
        gaussian = losses.BiVariateGaussian.from_scalars(mu, sd, ro)
        x1, x2 = (gaussian.to_tensor(x1), gaussian.to_tensor(x2))
        density = gaussian.density(x1, x2).item()
        self.assertAlmostEqual(density, 1.0 / (2 * math.pi), places=6)


class MixtureTests(unittest.TestCase):
    def test_num_components(self):
        pi = torch.zeros(1, 3)
        mu = torch.tensor([[10, 11, 12, 20, 21, 22]])
        sd = torch.zeros(1, 6)
        ro = torch.zeros(1, 3)
        mixture = losses.Mixture(pi, mu, sd, ro)

        self.assertEqual(mixture.num_components, 3)

    def test_mu1_and_mu2(self):
        pi = torch.zeros(2, 3)
        mu = torch.tensor([[10, 11, 12, 20, 21, 22], [1, 2, 3, 101, 102, 103]])
        sd = torch.zeros(2, 6)
        ro = torch.zeros(2, 3)
        mixture = losses.Mixture(pi, mu, sd, ro)

        expected_mu1 = torch.tensor([[10, 11, 12], [1, 2, 3]])
        expected_mu2 = torch.tensor([[20, 21, 22], [101, 102, 103]])

        self.assertTrue(torch.allclose(mixture.mu1, expected_mu1))
        self.assertTrue(torch.allclose(mixture.mu2, expected_mu2))

    def test_sd1_and_sd2(self):
        pi = torch.zeros(2, 3)
        mu = torch.zeros(2, 6)
        sd = torch.tensor([[10, 11, 12, 20, 21, 22], [1, 2, 3, 101, 102, 103]])
        ro = torch.zeros(2, 3)
        mixture = losses.Mixture(pi, mu, sd, ro)

        expected_sd1 = torch.tensor([[10, 11, 12], [1, 2, 3]])
        expected_sd2 = torch.tensor([[20, 21, 22], [101, 102, 103]])

        self.assertTrue(torch.allclose(mixture.sd1, expected_sd1))
        self.assertTrue(torch.allclose(mixture.sd2, expected_sd2))

    def test_1_component_mixture_density_for_a_single_step(self):
        pi = torch.ones(1, 1, dtype=torch.float64)
        mu = torch.tensor([[2, 3]], dtype=torch.float64)
        sd = torch.tensor([[3, 4]], dtype=torch.float64)
        ro = torch.tensor([[0.5]], dtype=torch.float64)

        x1 = torch.tensor([-1], dtype=torch.float64)
        x2 = torch.tensor([1], dtype=torch.float64)

        mixture = losses.Mixture(pi, mu, sd, ro)

        density = mixture.log_density(x1, x2)
        self.assertTupleEqual(density.shape, tuple())

        gaussian = losses.BiVariateGaussian.from_scalars(mu=(2., 3.), sd=(3., 4.), ro=0.5)
        d = torch.log(gaussian.density(-1, 1))
        self.assertAlmostEqual(density.item(), d.item(), places=6)

    def test_2_component_mixture_density_for_a_single_step(self):
        pi = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
        mu = torch.tensor([[2, -2, 3, -3]], dtype=torch.float64)
        sd = torch.tensor([[3, 3, 4, 4]], dtype=torch.float64)
        ro = torch.tensor([[0.5, -0.25]], dtype=torch.float64)

        x1 = torch.tensor([-1], dtype=torch.float64)
        x2 = torch.tensor([1], dtype=torch.float64)

        mixture = losses.Mixture(pi, mu, sd, ro)

        density = mixture.log_density(x1, x2)
        self.assertTupleEqual(density.shape, tuple())

        first_component_gaussian = losses.BiVariateGaussian.from_scalars(mu=(2., 3.), sd=(3., 4.), ro=0.5)
        second_component_gaussian = losses.BiVariateGaussian.from_scalars(mu=(-2., -3.), sd=(-3., -4.), ro=-0.25)

        mixture_density = (pi[0, 0] * first_component_gaussian.density(-1, 1) +
                           pi[0, 1] * second_component_gaussian.density(-1, 1))
        expected = torch.log(mixture_density).item()
        self.assertAlmostEqual(density.item(), expected, places=6)

    def test_1_component_mixture_density_for_2_steps_sequence(self):
        pi = torch.tensor([[1], [1]], dtype=torch.float64)
        mu = torch.tensor([[2, -2], [3, -3]], dtype=torch.float64)
        sd = torch.tensor([[3, 3], [4, 4]], dtype=torch.float64)
        ro = torch.tensor([[0.5], [-0.25]], dtype=torch.float64)

        x1 = torch.tensor([-1, 3], dtype=torch.float64)
        x2 = torch.tensor([1, 2], dtype=torch.float64)

        mixture = losses.Mixture(pi, mu, sd, ro)

        density = mixture.log_density(x1, x2)
        self.assertTupleEqual(density.shape, tuple())

        first_step_gaussian = losses.BiVariateGaussian.from_scalars(mu=(2., -2.), sd=(3., 3.), ro=0.5)
        second_step_gaussian = losses.BiVariateGaussian.from_scalars(mu=(3., -3.), sd=(4., 4.), ro=-0.25)

        expected = (torch.log(first_step_gaussian.density(-1, 1)) +
                    torch.log(second_step_gaussian.density(3, 2)))
        self.assertAlmostEqual(density.item(), expected.item(), places=6)

    def test_2_component_mixture_density_for_2_steps_sequence(self):
        pi = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float64)
        mu = torch.tensor([[2, 0, -2, 1], [3, 1, -3, 0]], dtype=torch.float64)
        sd = torch.tensor([[3, 1, 3, 5], [4, 2, 4, 6]], dtype=torch.float64)
        ro = torch.tensor([[0.5, 0.1], [-0.25, 0.3]], dtype=torch.float64)

        x1 = torch.tensor([-1, 3], dtype=torch.float64)
        x2 = torch.tensor([1, 2], dtype=torch.float64)

        mixture = losses.Mixture(pi, mu, sd, ro)

        density = mixture.log_density(x1, x2)
        self.assertTupleEqual(density.shape, tuple())

        step1_comp1_gaussian = losses.BiVariateGaussian.from_scalars(mu=(2., -2.), sd=(3., 3.), ro=0.5)
        step1_comp2_gaussian = losses.BiVariateGaussian.from_scalars(mu=(0., 1.), sd=(1., 5.), ro=0.1)

        step2_comp1_gaussian = losses.BiVariateGaussian.from_scalars(mu=(3., -3.), sd=(4., 4.), ro=-0.25)
        step2_comp2_gaussian = losses.BiVariateGaussian.from_scalars(mu=(1., 0.), sd=(2., 6.), ro=0.3)

        step1_pi = pi[0]
        step2_pi = pi[1]

        step1_density = (step1_pi[0] * step1_comp1_gaussian.density(-1, 1) +
                         step1_pi[1] * step1_comp2_gaussian.density(-1, 1))

        step2_density = (step2_pi[0] * step2_comp1_gaussian.density(3, 2) +
                         step2_pi[1] * step2_comp2_gaussian.density(3, 2))

        expected = torch.log(step1_density) + torch.log(step2_density)
        self.assertAlmostEqual(density.item(), expected.item(), places=6)


class LossTests(unittest.TestCase):
    def test_loss(self):
        batch_size = 2
        sequence_size = 2
        num_components = 3

        pi = torch.tensor([
            [[0, 0.25, 0.75], [0, 0.5, 0.5]],
            [[0, 0.5, 0.5], [1, 0., 0.]],
        ], dtype=torch.float32)

        mu = torch.ones(batch_size, sequence_size, num_components * 2, dtype=torch.float32)
        sd = torch.ones(batch_size, sequence_size, num_components * 2, dtype=torch.float32)
        ro = torch.zeros(batch_size, sequence_size, num_components, dtype=torch.float32)

        eos_hat = torch.zeros(batch_size, sequence_size, 1)
        ground_true = torch.zeros(batch_size, sequence_size, 3).numpy().tolist()
        ground_true = utils.PaddedSequencesBatch(ground_true)
        mixtures = (pi, mu, sd, ro)
        nll_loss = losses.nll_loss(mixtures, eos_hat, ground_true)
        nll_loss.item()


# todo: numerical stability tests
