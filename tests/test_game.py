import numpy as np
import pytest
from src.game import *


class TestUtilities:
    def test_active_higher_cost(self):
        """Active utility < inactive utility when cost is high."""
        k, R, L, B, c = 10, 10.0, 100.0, 10.0, 100.0
        u_active = utility_active(k, R, L, B, c)
        u_inactive = utility_inactive(k - 1, R, L, B)
        assert u_active < u_inactive


class TestNashEquilibrium:
    def test_formula_matches_search(self):
        """Analytical NE matches exhaustive search."""
        for N in [50, 100, 200]:
            for c in [0.1, 1.0, 5.0]:
                k_formula = find_nash_equilibrium(N, 10.0, 100.0, 10.0, c)
                k_search = find_nash_equilibrium_search(N, 10.0, 100.0, 10.0, c)
                assert k_formula == k_search, f"Mismatch at N={N}, c={c}"

    def test_ne_is_equilibrium(self):
        """Verify NE satisfies equilibrium conditions."""
        N, R, L, B, c = 100, 10.0, 100.0, 10.0, 1.0
        k_star = find_nash_equilibrium(N, R, L, B, c)

        if k_star > 0:
            u_active = utility_active(k_star, R, L, B, c)
            u_deviate = utility_inactive(k_star - 1, R, L, B)
            assert u_active >= u_deviate - 1e-9

        if k_star < N:
            u_inactive = utility_inactive(k_star, R, L, B)
            u_deviate = utility_active(k_star + 1, R, L, B, c)
            assert u_inactive >= u_deviate - 1e-9


class TestSocialOptimum:
    def test_formula_matches_search(self):
        """Analytical optimum matches exhaustive search."""
        for N in [50, 100, 200]:
            for c in [0.1, 1.0, 5.0]:
                k_formula = find_social_optimum(N, 10.0, 100.0, 10.0, c)
                k_search = find_social_optimum_search(N, 10.0, 100.0, 10.0, c)
                assert k_formula == k_search, f"Mismatch at N={N}, c={c}"

    def test_optimum_geq_ne(self):
        """Social optimum >= Nash equilibrium."""
        for N in [50, 100, 200]:
            for c in [0.1, 1.0, 5.0]:
                k_star = find_nash_equilibrium(N, 10.0, 100.0, 10.0, c)
                k_opt = find_social_optimum(N, 10.0, 100.0, 10.0, c)
                assert k_opt >= k_star


class TestPoA:
    def test_poa_geq_one(self):
        """PoA >= 1 always."""
        for N in [50, 100]:
            for c in [0.5, 1.0, 2.0]:
                poa = price_of_anarchy(N, 10.0, 100.0, 10.0, c)
                assert poa >= 1.0 - 1e-9
