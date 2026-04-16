import math
import unittest
from typing import Any, List

from logic.src.policies.route_construction.acceptance_criteria.skewed_variable_neighborhood_search import SkewedVNSAcceptance
from logic.src.policies.route_construction.acceptance_criteria.epsilon_dominance import EpsilonDominanceCriterion
from logic.src.policies.route_construction.acceptance_criteria.adaptive_boltzmann_metropolis import AdaptiveBoltzmannMetropolis
from logic.src.interfaces.distance_metric import IDistanceMetric


class MockDistanceMetric(IDistanceMetric):
    def compute(self, current: Any, candidate: Any) -> float:
        # Simple absolute difference for testing
        return float(abs(current - candidate))


class TestAdvancedCriteria(unittest.TestCase):
    def test_skewed_vns_maximization(self):
        metric = MockDistanceMetric()
        # alpha = 0.5. For max: accept if f_cand > f_cur - 0.5 * rho
        criterion = SkewedVNSAcceptance(alpha=0.5, metric=metric, maximization=True)

        # 1. Improving move: always accept
        self.assertTrue(criterion.accept(10.0, 12.0)[0])

        # 2. Worsening move but structurally distant
        # f_cur = 12, f_cand = 10, dist = abs(12-10) = 2
        # threshold = 12 - 0.5 * 2 = 11
        # 10 > 11 is False -> Reject
        self.assertFalse(criterion.accept(12.0, 10.0, current_sol=12, candidate_sol=10)[0])

        # 3. Worsening move with high distance
        # f_cur = 12, f_cand = 10, dist = 10
        # threshold = 12 - 0.5 * 10 = 7
        # 10 > 7 is True -> Accept (Skewed!)
        self.assertTrue(criterion.accept(12.0, 10.0, current_sol=12, candidate_sol=22)[0])

    def test_epsilon_dominance_maximization(self):
        # epsilon = 1.0. Grid box = floor(obj / 1.0)
        criterion = EpsilonDominanceCriterion(epsilon=1.0, maximization=True)

        # First solution: (10.5, 20.5) -> Box (10, 20)
        criterion.step(0.0, (10.5, 20.5), accepted=True)

        # Candidate inside same box: (10.9, 20.1) -> Box (10, 20) -> Reject
        self.assertFalse(criterion.accept((10.5, 20.5), (10.9, 20.1))[0])

        # Candidate in better box: (11.1, 20.5) -> Box (11, 20) -> Accept
        self.assertTrue(criterion.accept((10.5, 20.5), (11.1, 20.5))[0])

        # Candidate in dominated box: (9.5, 19.5) -> Box (9, 19) -> Reject
        self.assertFalse(criterion.accept((10.5, 20.5), (9.5, 19.5))[0])

    def test_adaptive_boltzmann_scaling(self):
        # p0 = 0.5 (chance of accepting 1-sigma move)
        criterion = AdaptiveBoltzmannMetropolis(p0=0.5, window_size=10, alpha=1.0, maximization=True)

        # Feed some transitions to stabilize sigma
        # deltas: 10, 10, 10, 10, 10 -> sigma = 0
        for _ in range(5):
            criterion.step(100.0, 110.0, accepted=True)

        self.assertEqual(len(criterion.deltas), 5)
        self.assertEqual(criterion.sigma, 0.0)

        # Feed variation: 0, 10, 0, 10, ...
        for i in range(5):
            val = 110.0 if i % 2 == 0 else 100.0
            criterion.step(100.0, val, accepted=True)

        self.assertGreater(criterion.sigma, 0.0)
        self.assertGreater(criterion.temp, 0.0)

        # Verify p0 logic: at T_0, accepting a delta=-sigma move should be around p0=0.5
        # Prob = exp(-sigma / T) = exp(-sigma / (-sigma / ln(p0))) = exp(ln(p0)) = p0
        sigma = criterion.sigma
        temp = criterion.temp
        delta = -sigma
        prob = math.exp(delta / temp)
        self.assertAlmostEqual(prob, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
