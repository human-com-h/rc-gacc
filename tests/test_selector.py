"""Independent exhaustive checks of the optimizer and degeneration boundary."""
import itertools
import unittest
import numpy as np
from rc_gacc.selector import Edge, solve_milp, objective, pair_loss


class SolverTests(unittest.TestCase):
    def test_exhaustive_optimum_with_coverage_and_closure(self):
        rng = np.random.default_rng(42)
        for repeat in range(12):
            n = 7
            costs = rng.integers(2, 10, n)
            u = rng.uniform(0, 1, n)
            a = rng.uniform(0, 1, (3, n))
            edges = [Edge(0, 1, "C"), Edge(2, 3, "Q"), Edge(3, 4, "Q")]
            selected, log = solve_milp(costs, 18, u, a, edges)
            values = []
            for bits in itertools.product((0, 1), repeat=n):
                if np.dot(bits, costs) <= 18 and bits[0] == bits[1] and bits[2] <= bits[3] <= bits[4]:
                    s = [i for i,b in enumerate(bits) if b]
                    values.append(objective(s,u,a,edges))
            self.assertTrue(log["proven_optimal"])
            self.assertAlmostEqual(objective(selected,u,a,edges), max(values), places=6)

    def test_unaffordable_pair_drops_both_without_breaking_budget(self):
        s, log = solve_milp([11,11,2],10,[2,2,0.1],np.zeros((0,3)),[Edge(0,1,"C")])
        self.assertEqual(s,[2])
        self.assertFalse(log["violations"])

    def test_optional_slack_is_explicit_and_budget_remains_hard(self):
        s, log = solve_milp([7,7],8,[10,1],np.zeros((0,2)),[Edge(0,1,"C")],relax_conflicts=True)
        self.assertEqual(s,[0])
        self.assertEqual(len(log["violations"]),1)
        self.assertLessEqual(sum([7,7][i] for i in s),8)

    def test_direction_does_not_force_reverse(self):
        s,_=solve_milp([10,2],3,[10,1],np.zeros((0,2)),[Edge(0,1,"Q")])
        self.assertEqual(s,[1])

    def test_anti_double_drop_and_frozen_gate(self):
        e=[Edge(0,1,"C")]
        self.assertGreater(pair_loss([0,0],[1,1],e),0)
        self.assertEqual(pair_loss([0.7,0.7],[1,1],e),0)
        self.assertEqual(pair_loss([0,0],[0.1,0.1],e),0)

if __name__ == "__main__":
    unittest.main()
