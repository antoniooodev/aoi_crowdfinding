"""Unit tests for `src.aoi`.

Focuses on edge cases (P_det in {0,1}), monotonicity in k, and basic trajectory invariants.
"""

import numpy as np
import pytest

from src.aoi import *


class TestExpectedAoI:
    def test_zero_detection(self):
        assert expected_aoi(0.0) == np.inf

    def test_certain_detection(self):
        assert expected_aoi(1.0) == 0.0

    def test_half_detection(self):
        assert expected_aoi(0.5) == 1.0

    def test_formula(self):
        P = 0.3
        expected = 1.0 / P - 1.0
        assert abs(expected_aoi(P) - expected) < 1e-10


class TestAoIFromK:
    def test_monotonicity(self):
        R, L = 10.0, 100.0
        aoi_values = [expected_aoi_from_k(k, R, L) for k in range(1, 101)]
        assert all(aoi_values[i] >= aoi_values[i + 1] for i in range(99))

    def test_zero_volunteers(self):
        assert expected_aoi_from_k(0, 10.0, 100.0) == np.inf


class TestTrajectory:
    def test_shape(self):
        traj = simulate_aoi_trajectory(0.5, 1000, seed=42)
        assert traj.shape == (1000,)

    def test_non_negative(self):
        traj = simulate_aoi_trajectory(0.5, 1000, seed=42)
        assert np.all(traj >= 0)

    def test_convergence(self):
        """Test that simulated average converges to theoretical."""
        P_det = 0.3
        T = 100000
        traj = simulate_aoi_trajectory_fast(P_det, T, seed=42)
        simulated = time_average_aoi(traj)
        theoretical = expected_aoi(P_det)
        assert abs(simulated - theoretical) / theoretical < 0.05


class TestFastVsSlow:
    def test_equivalence(self):
        """Test that fast and slow implementations give same statistics."""
        P_det = 0.4
        T = 10000

        traj_slow = simulate_aoi_trajectory(P_det, T, seed=42)
        traj_fast = simulate_aoi_trajectory_fast(P_det, T, seed=42)

        # Same trajectory with same seed
        np.testing.assert_array_equal(traj_slow, traj_fast)
