import unittest
import math
from logic.src.policies.route_construction.acceptance_criteria.demon_algorithm import DemonAlgorithm
from logic.src.policies.route_construction.acceptance_criteria.generalized_tsallis_sa import GeneralizedTsallisSA
from logic.src.policies.route_construction.acceptance_criteria.non_linear_great_deluge import NonLinearGreatDeluge
from logic.src.policies.route_construction.acceptance_criteria.exponential_monte_carlo_counter import EMCQAcceptance


class TestAdvancedAlgorithms(unittest.TestCase):

    def test_demon_algorithm_minimization(self):
        # 5 warm-up steps. D0 will be max of history.
        criterion = DemonAlgorithm(warm_up_steps=3, maximization=False)

        # Warm-up phase
        criterion.step(100, 95, accepted=True)  # delta = -5, history=[5]
        criterion.step(95, 92, accepted=True)   # delta = -3, history=[5, 3]
        criterion.step(92, 82, accepted=True)   # delta = -10, history=[5, 3, 10]
        # D0 should now be 10.
        self.assertEqual(criterion.demon_credit, 10.0)
        self.assertTrue(criterion._warmed_up)

        # 1. Accept worsening within credit
        self.assertTrue(criterion.accept(82, 87)) # delta = 5 <= 10
        criterion.step(82, 87, accepted=True)
        self.assertEqual(criterion.demon_credit, 5.0)

        # 2. Reject worsening exceeding credit
        self.assertFalse(criterion.accept(87, 95)) # delta = 8 > 5

        # 3. Recharge on improvement
        criterion.step(87, 80, accepted=True) # delta = -7
        self.assertEqual(criterion.demon_credit, 7.0)

    def test_generalized_tsallis_sa(self):
        # q = 1.5. Converges to Boltzmann-Gibbs at q=1.
        criterion = GeneralizedTsallisSA(q=1.5, p0=0.5, window_size=5, maximization=False)

        # Feed some transitions to initialize T0
        for _ in range(5):
            criterion.step(100, 110, accepted=False) # deltas = 10

        self.assertGreater(criterion.temp, 0.0)
        self.assertGreater(criterion.sigma, 0.0)

        # Verify p0 logic: at T0, delta=sigma should yield p0
        # Prob = [1 - (1-q)*(sigma/T)]^(1/(1-q))
        sigma = criterion.sigma
        temp = criterion.temp
        delta = sigma
        q = criterion.q

        term = 1.0 - (1.0 - q) * (delta / temp)
        prob = math.pow(term, 1.0 / (1.0 - q))
        self.assertAlmostEqual(prob, 0.5, places=5)

    def test_non_linear_great_deluge_minimization(self):
        # t_max = 100, beta = 5.0
        criterion = NonLinearGreatDeluge(t_max=100, initial_tolerance=0.1, gap_epsilon=0.01, beta=5.0, maximization=False)
        criterion.setup(100.0) # Level0 = 110.0

        self.assertEqual(criterion.water_level, 110.0)

        # Progress 50%
        for _ in range(50):
            criterion.step(100.0, 95.0, accepted=True, f_best=90.0)

        # f_target = 90 * (1 - 0.01) = 89.1
        # Level = 89.1 + (110 - 89.1) * exp(-5 * 0.5)
        # 89.1 + 20.9 * 0.082 = 89.1 + 1.71 = 90.81
        self.assertAlmostEqual(criterion.water_level, 90.8, places=1)

        # Check acceptance
        self.assertTrue(criterion.accept(95.0, 90.0, f_best=90.0)) # 90 <= Level
        self.assertFalse(criterion.accept(95.0, 92.0, f_best=90.0)) # 92 > Level

    def test_emcq_acceptance(self):
        # p=0, p_boost=1.0, Q=3
        criterion = EMCQAcceptance(p=0.0, p_boost=1.0, q_threshold=3, maximization=False)

        # Start with q=0
        self.assertFalse(criterion.accept(100, 110)) # p=0
        criterion.step(100, 110, accepted=False) # q=1

        self.assertFalse(criterion.accept(100, 110)) # p=0
        criterion.step(100, 110, accepted=False) # q=2

        self.assertFalse(criterion.accept(100, 110)) # p=0
        criterion.step(100, 110, accepted=False) # q=3

        # Now q >= Q, should accept!
        self.assertTrue(criterion.accept(100, 110)) # p_boost=1
        criterion.step(100, 110, accepted=True) # q resets to 0
        self.assertEqual(criterion.rejection_counter, 0)


if __name__ == "__main__":
    unittest.main()
