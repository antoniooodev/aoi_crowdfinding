import numpy as np

from src.game import find_nash_equilibrium, find_social_optimum
from src.stackelberg import (
    induced_equilibrium,
    optimal_incentive_for_target,
    optimal_incentive,
    total_incentive_cost,
    platform_objective,
)


class TestInducedEquilibrium:
    def test_matches_cost_shift(self) -> None:
        N, R, L, B, c = 100, 10.0, 100.0, 10.0, 25.0
        p = 3.0

        k1 = induced_equilibrium(p, N, R, L, B, c)
        k2 = find_nash_equilibrium(N, R, L, B, max(0.0, c - p))
        assert k1 == k2


class TestOptimalIncentive:
    def test_optimal_incentive_for_target_formula(self) -> None:
        R, L, B, c = 10.0, 100.0, 10.0, 25.0
        k_target = 8

        rho = np.pi * R**2 / L**2
        expected = max(0.0, c - (B * rho * (1.0 - rho) ** (k_target - 1)))
        got = optimal_incentive_for_target(k_target, R, L, B, c)

        assert abs(got - expected) < 1e-12

    def test_implements_social_optimum(self) -> None:
        N, R, L, B, c = 100, 10.0, 100.0, 10.0, 25.0

        k_opt = find_social_optimum(N, R, L, B, c)
        p_star = optimal_incentive(N, R, L, B, c)

        k_induced = induced_equilibrium(p_star, N, R, L, B, c)
        assert k_induced == k_opt


class TestCostsAndObjective:
    def test_total_incentive_cost(self) -> None:
        N, R, L, B, c = 100, 10.0, 100.0, 10.0, 25.0
        p = 5.0

        k = induced_equilibrium(p, N, R, L, B, c)
        assert total_incentive_cost(p, N, R, L, B, c) == p * k

    def test_budget_constraint(self) -> None:
        N, R, L, B, c = 100, 10.0, 100.0, 10.0, 25.0
        p = 5.0
        k = induced_equilibrium(p, N, R, L, B, c)

        too_small_budget = p * k - 1e-9
        ok_budget = p * k + 1e-9

        assert platform_objective(p, N, R, L, B, c, budget=too_small_budget) == float("-inf")
        assert platform_objective(p, N, R, L, B, c, budget=ok_budget) != float("-inf")
