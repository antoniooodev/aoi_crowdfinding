"""
test_heterogeneous.py - Unit tests for heterogeneous cost model

Tests:
1. HeterogeneousCostParams: CDF, PDF, quantile, sampling
2. Nash equilibrium: corner cases, interior equilibrium, monotonicity
3. Social optimum: greedy selection, order statistics
4. Price of Anarchy: bounds, comparison with homogeneous
5. Stackelberg: incentive correctness, budget constraints
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HeterogeneousCostParams
from src.game import (
    find_nash_equilibrium,
    find_social_optimum,
    price_of_anarchy,
    find_nash_heterogeneous,
    find_social_optimum_heterogeneous,
    find_social_optimum_heterogeneous_expected,
    price_of_anarchy_heterogeneous,
    expected_welfare_heterogeneous,
    nash_threshold,
    analyze_equilibrium_heterogeneous,
)
from src.stackelberg import (
    analyze_stackelberg_heterogeneous,
    optimal_incentive_heterogeneous,
    induced_equilibrium_heterogeneous,
)
from src.simulation import (
    simulate_threshold_equilibrium,
    validate_heterogeneous_model,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def default_params():
    """Default test parameters."""
    return {
        'N': 100,
        'R': 30.0,
        'L': 500.0,
        'B': 10.0,
        'rho': np.pi * 30.0**2 / 500.0**2,  # ≈ 0.0113
    }


@pytest.fixture
def uniform_cost_params():
    """Uniform cost distribution."""
    return HeterogeneousCostParams(c_min=0.02, c_max=0.08, distribution="uniform")


@pytest.fixture
def homogeneous_cost_params():
    """Degenerate (homogeneous) distribution."""
    return HeterogeneousCostParams(c_min=0.05, c_max=0.05, distribution="uniform")


# =============================================================================
# TEST: HeterogeneousCostParams
# =============================================================================

class TestHeterogeneousCostParams:
    """Tests for cost distribution parameters."""
    
    def test_uniform_cdf_bounds(self, uniform_cost_params):
        """CDF should be 0 at c_min, 1 at c_max."""
        cp = uniform_cost_params
        assert cp.cdf(cp.c_min) == 0.0
        assert cp.cdf(cp.c_max) == 1.0
        assert cp.cdf(cp.c_min - 0.01) == 0.0
        assert cp.cdf(cp.c_max + 0.01) == 1.0
    
    def test_uniform_cdf_midpoint(self, uniform_cost_params):
        """CDF at midpoint should be 0.5."""
        cp = uniform_cost_params
        midpoint = (cp.c_min + cp.c_max) / 2
        assert abs(cp.cdf(midpoint) - 0.5) < 1e-10
    
    def test_uniform_quantile_inverse(self, uniform_cost_params):
        """Quantile should be inverse of CDF."""
        cp = uniform_cost_params
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            c = cp.quantile(q)
            assert abs(cp.cdf(c) - q) < 1e-10
    
    def test_uniform_mean_cost(self, uniform_cost_params):
        """Mean cost should be midpoint for uniform."""
        cp = uniform_cost_params
        expected_mean = (cp.c_min + cp.c_max) / 2
        assert abs(cp.mean_cost - expected_mean) < 1e-10
    
    def test_sampling_bounds(self, uniform_cost_params):
        """Samples should be within [c_min, c_max]."""
        cp = uniform_cost_params
        rng = np.random.default_rng(42)
        samples = cp.sample(1000, rng)
        assert np.all(samples >= cp.c_min)
        assert np.all(samples <= cp.c_max)
    
    def test_sampling_mean(self, uniform_cost_params):
        """Sample mean should approximate theoretical mean."""
        cp = uniform_cost_params
        rng = np.random.default_rng(42)
        samples = cp.sample(10000, rng)
        assert abs(np.mean(samples) - cp.mean_cost) < 0.001
    
    def test_homogeneous_detection(self, homogeneous_cost_params):
        """Should detect homogeneous (degenerate) distribution."""
        assert homogeneous_cost_params.is_homogeneous
    
    def test_heterogeneous_detection(self, uniform_cost_params):
        """Should detect heterogeneous distribution."""
        assert not uniform_cost_params.is_homogeneous
    
    def test_order_statistic_bounds(self, uniform_cost_params):
        """Order statistics should be increasing."""
        cp = uniform_cost_params
        N = 100
        prev = cp.c_min
        for k in range(1, N + 1):
            expected_k = cp.expected_kth_order_statistic(k, N)
            assert expected_k >= prev
            prev = expected_k
        assert prev <= cp.c_max + 1e-10
    
    def test_sum_of_k_lowest_increasing(self, uniform_cost_params):
        """Sum of k lowest should increase with k."""
        cp = uniform_cost_params
        N = 100
        prev_sum = 0
        for k in range(1, N + 1):
            current_sum = cp.expected_sum_of_k_lowest(k, N)
            assert current_sum >= prev_sum
            prev_sum = current_sum


# =============================================================================
# TEST: Nash Equilibrium
# =============================================================================

class TestNashEquilibrium:
    """Tests for Nash equilibrium computation."""
    
    def test_homogeneous_equivalence(self, default_params, homogeneous_cost_params):
        """Heterogeneous with c_min=c_max should match homogeneous."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        c = homogeneous_cost_params.c_min
        
        k_hom = find_nash_equilibrium(N, default_params['R'], default_params['L'], B, c)
        k_het, _ = find_nash_heterogeneous(N, rho, B, homogeneous_cost_params)
        
        assert abs(k_hom - k_het) <= 1  # Allow rounding difference
    
    def test_corner_case_all_participate(self, default_params):
        """When c_max is very low, everyone should participate."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        
        # c_max << Bρ(1-ρ)^(N-1)
        cp = HeterogeneousCostParams(c_min=0.001, c_max=0.002)
        k_star, _ = find_nash_heterogeneous(N, rho, B, cp)
        
        assert k_star == N
    
    def test_corner_case_none_participate(self, default_params):
        """When c_min > Bρ, no one should participate."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        B_rho = B * rho
        
        # c_min > Bρ
        cp = HeterogeneousCostParams(c_min=B_rho + 0.01, c_max=B_rho + 0.02)
        k_star, _ = find_nash_heterogeneous(N, rho, B, cp)
        
        assert k_star == 0
    
    def test_monotonicity_in_mean_cost(self, default_params):
        """k* should decrease as mean cost increases."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        
        k_values = []
        for mean_c in np.linspace(0.02, 0.10, 10):
            cp = HeterogeneousCostParams(c_min=mean_c * 0.5, c_max=mean_c * 1.5)
            k_star, _ = find_nash_heterogeneous(N, rho, B, cp)
            k_values.append(k_star)
        
        # Should be non-increasing
        for i in range(len(k_values) - 1):
            assert k_values[i] >= k_values[i + 1]
    
    def test_equilibrium_consistency(self, default_params, uniform_cost_params):
        """k* should satisfy fixed-point condition approximately."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        cp = uniform_cost_params
        
        k_star, c_bar_star = find_nash_heterogeneous(N, rho, B, cp)
        
        # Check: k* ≈ N * F(c̄(k*))
        c_bar_check = nash_threshold(k_star, rho, B)
        k_implied = N * cp.cdf(c_bar_check)
        
        assert abs(k_star - k_implied) <= 2  # Allow small discrepancy


# =============================================================================
# TEST: Social Optimum
# =============================================================================

class TestSocialOptimum:
    """Tests for social optimum computation."""
    
    def test_optimum_geq_nash(self, default_params, uniform_cost_params):
        """k_opt should be >= k* (positive externality)."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        
        k_star, _ = find_nash_heterogeneous(N, rho, B, uniform_cost_params)
        k_opt, _ = find_social_optimum_heterogeneous_expected(N, rho, B, uniform_cost_params)
        
        assert k_opt >= k_star
    
    def test_optimum_monotonicity(self, default_params):
        """k_opt should decrease as mean cost increases."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        
        k_values = []
        for mean_c in np.linspace(0.5, 5.0, 10):
            cp = HeterogeneousCostParams(c_min=mean_c * 0.5, c_max=mean_c * 1.5)
            k_opt, _ = find_social_optimum_heterogeneous_expected(N, rho, B, cp)
            k_values.append(k_opt)
        
        # Should be non-increasing
        for i in range(len(k_values) - 1):
            assert k_values[i] >= k_values[i + 1]
    
    def test_greedy_selection_optimality(self, default_params, uniform_cost_params):
        """Greedy selection should maximize welfare."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        cp = uniform_cost_params
        
        rng = np.random.default_rng(42)
        costs = cp.sample(N, rng)
        
        k_opt, active_opt, welfare_opt = find_social_optimum_heterogeneous(N, rho, B, costs)
        
        # Check that adding one more would decrease welfare
        if k_opt < N:
            sorted_costs = np.sort(costs)
            next_cost = sorted_costs[k_opt]
            marginal_benefit = N * B * rho * (1 - rho) ** k_opt
            assert next_cost > marginal_benefit  # Adding k_opt+1 is not beneficial


# =============================================================================
# TEST: Price of Anarchy
# =============================================================================

class TestPriceOfAnarchy:
    """Tests for Price of Anarchy computation."""
    
    def test_poa_geq_one(self, default_params, uniform_cost_params):
        """PoA should be >= 1."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        
        poa = price_of_anarchy_heterogeneous(N, rho, B, uniform_cost_params)
        
        assert poa >= 1.0 or np.isinf(poa)
    
    def test_poa_homogeneous_comparison(self, default_params):
        """PoA should be similar for homogeneous and low-heterogeneity cases."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        R, L = default_params['R'], default_params['L']
        c = 0.05
        
        poa_hom = price_of_anarchy(N, R, L, B, c)
        
        cp_low_het = HeterogeneousCostParams(c_min=0.048, c_max=0.052)  # Low spread
        poa_het = price_of_anarchy_heterogeneous(N, rho, B, cp_low_het)
        
        # Should be close
        if np.isfinite(poa_hom) and np.isfinite(poa_het):
            assert abs(poa_hom - poa_het) / poa_hom < 0.2  # Within 20%


# =============================================================================
# TEST: Stackelberg Incentives
# =============================================================================

class TestStackelberg:
    """Tests for Stackelberg incentive mechanism."""
    
    def test_incentive_increases_participation(self, default_params, uniform_cost_params):
        """Positive incentive should increase participation."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        cp = uniform_cost_params
        
        k_no_incentive, _ = find_nash_heterogeneous(N, rho, B, cp)
        k_with_incentive, _ = induced_equilibrium_heterogeneous(0.02, N, rho, B, cp)
        
        assert k_with_incentive >= k_no_incentive
    
    def test_optimal_incentive_achieves_optimum(self, default_params, uniform_cost_params):
        """Optimal incentive should achieve k_opt."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        cp = uniform_cost_params
        
        k_opt, _ = find_social_optimum_heterogeneous_expected(N, rho, B, cp)
        p_star, k_induced, _ = optimal_incentive_heterogeneous(N, rho, B, cp)
        
        # Should be close to k_opt
        assert abs(k_induced - k_opt) <= 2
    
    def test_zero_incentive_when_optimal(self, default_params):
        """If k* = k_opt, no incentive needed."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        
        # Very low cost: everyone participates at both Nash and optimum
        cp = HeterogeneousCostParams(c_min=0.001, c_max=0.002)
        
        p_star, _, _ = optimal_incentive_heterogeneous(N, rho, B, cp)
        
        assert p_star < 0.001  # Essentially zero


# =============================================================================
# TEST: Monte Carlo Validation
# =============================================================================

class TestMonteCarloValidation:
    """Tests for Monte Carlo validation functions."""
    
    def test_threshold_equilibrium_validation(self, default_params, uniform_cost_params):
        """MC validation should confirm analytical predictions."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        
        results = simulate_threshold_equilibrium(N, rho, B, uniform_cost_params, 
                                                  n_runs=200, seed=42)
        
        # Relative error should be small
        assert results['k_star_relative_error'] < 0.15  # Within 15%
    
    def test_full_validation_suite(self, default_params, uniform_cost_params):
        """Full validation should pass."""
        N, rho, B = default_params['N'], default_params['rho'], default_params['B']
        
        results = validate_heterogeneous_model(N, rho, B, uniform_cost_params,
                                                n_runs=100, seed=42, tolerance=0.20)
        
        # At least Nash should validate
        assert results['nash_validated']


# =============================================================================
# TEST: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_volunteer(self):
        """Should handle N=1 correctly."""
        N, rho, B = 1, 0.01, 10.0
        cp = HeterogeneousCostParams(c_min=0.05, c_max=0.15)
        
        k_star, _ = find_nash_heterogeneous(N, rho, B, cp)
        k_opt, _ = find_social_optimum_heterogeneous_expected(N, rho, B, cp)
        
        assert k_star in [0, 1]
        assert k_opt in [0, 1]
    
    def test_very_small_rho(self):
        """Should handle very small coverage probability."""
        N, rho, B = 100, 0.0001, 10.0
        cp = HeterogeneousCostParams(c_min=0.0001, c_max=0.0002)
        
        k_star, _ = find_nash_heterogeneous(N, rho, B, cp)
        
        # Should not crash
        assert k_star >= 0
        assert k_star <= N
    
    def test_very_large_N(self):
        """Should handle large N without overflow."""
        N, rho, B = 10000, 0.01, 10.0
        cp = HeterogeneousCostParams(c_min=0.05, c_max=0.15)
        
        k_star, _ = find_nash_heterogeneous(N, rho, B, cp)
        k_opt, _ = find_social_optimum_heterogeneous_expected(N, rho, B, cp)
        
        assert np.isfinite(k_star)
        assert np.isfinite(k_opt)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])