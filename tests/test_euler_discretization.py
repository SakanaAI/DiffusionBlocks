"""Exact checks for docs/euler_discretization.md, not a model benchmark."""
from fractions import Fraction as Q
import unittest


def documented_euler_step(z, denoised, sigma, next_sigma):
    """The scalar form of the existing signed-step sampler expression."""
    return z + (next_sigma - sigma) * (z - denoised) / sigma


class EulerDiscretizationTests(unittest.TestCase):
    def test_descending_step_moves_toward_fixed_target(self):
        self.assertEqual(documented_euler_step(Q(1), Q(0), Q(1), Q(1, 2)), Q(1, 2))

    def test_opposite_sign_is_not_the_same_step(self):
        z, denoised, sigma, next_sigma = Q(1), Q(0), Q(1), Q(1, 2)
        positive_delta = sigma - next_sigma
        printed = z + positive_delta * (z - denoised) / sigma
        self.assertEqual(printed, Q(3, 2))
        self.assertNotEqual(printed, documented_euler_step(z, denoised, sigma, next_sigma))

    def test_matches_euler_step_from_stated_ode(self):
        z, denoised, sigma, next_sigma = Q(7, 3), Q(-2, 5), Q(3, 2), Q(2, 3)
        score = (denoised - z) / sigma**2
        derivative = -sigma * score
        from_ode = z + (next_sigma - sigma) * derivative
        self.assertEqual(from_ode, documented_euler_step(z, denoised, sigma, next_sigma))

    def test_positive_delta_and_mixture_forms_match(self):
        for z in (Q(-2), Q(0), Q(5, 3)):
            for denoised in (Q(-1, 3), Q(0), Q(4)):
                sigma = Q(2)
                for next_sigma in (Q(0), Q(1, 2), Q(2)):
                    with self.subTest(z=z, denoised=denoised, next_sigma=next_sigma):
                        alpha = (sigma - next_sigma) / sigma
                        expected = documented_euler_step(z, denoised, sigma, next_sigma)
                        self.assertEqual(expected, z + alpha * (denoised - z))
                        self.assertEqual(expected, (1 - alpha) * z + alpha * denoised)

    def test_zero_step_preserves_state(self):
        self.assertEqual(documented_euler_step(Q(3), Q(8), Q(2), Q(2)), Q(3))

    def test_terminal_zero_step_reaches_fixed_denoiser(self):
        self.assertEqual(documented_euler_step(Q(3), Q(8), Q(2), Q(0)), Q(8))

    def test_fixed_point_is_preserved(self):
        self.assertEqual(documented_euler_step(Q(3), Q(3), Q(2), Q(1)), Q(3))

    def test_zero_based_loop_writes_final_state(self):
        sigmas = [Q(1), Q(1, 2), Q(0)]
        states = [Q(11)]
        accessed = []
        for i in range(len(sigmas) - 1):
            accessed.append(i)
            states.append(documented_euler_step(states[i], Q(7), sigmas[i], sigmas[i + 1]))
        self.assertEqual(accessed, [0, 1])
        self.assertEqual(states, [Q(11), Q(9), Q(7)])
        self.assertEqual(len(states), len(sigmas))


if __name__ == "__main__":
    unittest.main()
