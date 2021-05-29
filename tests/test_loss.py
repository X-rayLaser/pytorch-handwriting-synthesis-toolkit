import unittest
import torch
import loss
import math


class BiVariateNormalZTests(unittest.TestCase):
    def test_normalized_square_helper_methods(self):
        mu = (1., 4.)
        sd = (2., 4.)
        ro = 0.

        x1 = 4.
        gaussian = loss.BiVariateGaussian(mu, sd, ro)

        res = gaussian.normalized_square(x1, mu[0], sd[0])
        expected = ((x1 - mu[0]) / sd[0]) ** 2
        self.assertEqual(res, expected)
        self.assertEqual(res, 1.5 ** 2)

    def test_substraction_term_helper_method(self):
        mu = (1., 4.)
        sd = (2., 3.)
        ro = 0.5

        x1, x2 = (4., 5.)
        gaussian = loss.BiVariateGaussian(mu, sd, ro)

        res = gaussian.substraction_term(x1, x2)
        expected = 2 * ro * (x1 - mu[0]) * (x2 - mu[1]) / (sd[0] * sd[1])
        self.assertEqual(res, expected)
        self.assertEqual(res, 0.5)

    def test_formula_uses_first_term(self):
        mu = (1., 4.)
        sd = (2., 4.)
        ro = 0.

        x1 = 4.
        x2 = 4.
        gaussian = loss.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        expected = 9/4.0
        self.assertEqual(z, expected)
        self.assertEqual(z, (x1 - mu[0]) ** 2 / sd[0] ** 2)

    def test_formula_uses_second_term(self):
        mu = (4., 4.)
        sd = (2., 4.)
        ro = 0.

        x1, x2 = (4., 1.)
        gaussian = loss.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        expected = 9 / 16.
        self.assertEqual(z, expected)
        self.assertEqual(z, (x2 - mu[1]) ** 2 / sd[1] ** 2)

    def test_formula_uses_third_term(self):
        mu = (2., 4.)
        sd = (2., 3.)
        ro = 1

        x1, x2 = (4., 1.)
        gaussian = loss.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        expected = 1 + 1 - (-2)
        self.assertEqual(z, expected)
        expected = (gaussian.normalized_square(x1, mu[0], sd[0])
                    + gaussian.normalized_square(x2, mu[1], sd[1])
                    - 2 * ro * (x1 - mu[0]) * (x2 - mu[1]) / (sd[0] * sd[1]))
        self.assertEqual(z, expected)

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

        gaussian = loss.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        first_component = loss.BiVariateGaussian((2., 4.), (2., 3.), 1.)
        second_component = loss.BiVariateGaussian((5., 6.), (4., 3.), 0.5)

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

        gaussian = loss.BiVariateGaussian(mu, sd, ro)
        z = gaussian.compute_z(x1, x2)

        scalar_gaussian = loss.BiVariateGaussian(
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
        gaussian = loss.BiVariateGaussian.from_scalars(mu, sd, ro)
        density = gaussian.compute_density(torch.tensor(z))
        expected = 1.0 / (2 * math.pi * sd1 * sd2 * math.sqrt(1 - ro ** 2))
        self.assertEqual(density, expected)

    def test_compute_density_formula_correct(self):
        mu = (3., 4.)
        sd = (2., 3.)
        sd1, sd2 = sd
        ro = 0.5
        z = 3.2
        gaussian = loss.BiVariateGaussian.from_scalars(mu, sd, ro)
        density = gaussian.compute_density(gaussian.to_tensor(z))

        one_minus_ro_squared = 1 - ro ** 2
        exp = math.exp(-z / (2 * one_minus_ro_squared))
        denominator = (2 * math.pi * sd1 * sd2 * math.sqrt(1 - ro ** 2))
        expected = exp / denominator
        self.assertAlmostEqual(density.item(), expected, places=8)

    def test_density_on_scalar_inputs(self):
        mu = (3., 4.)
        sd = (2., 3.)
        ro = 0.5

        gaussian = loss.BiVariateGaussian.from_scalars(mu, sd, ro)
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

        gaussian = loss.BiVariateGaussian(mu, sd, ro)
        scalar_gaussian = loss.BiVariateGaussian.from_scalars(
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
        gaussian = loss.BiVariateGaussian.from_scalars(mu, sd, ro)
        x1, x2 = (gaussian.to_tensor(x1), gaussian.to_tensor(x2))
        density = gaussian.density(x1, x2)
        self.assertEqual(density, 1.0 / (2 * math.pi))
