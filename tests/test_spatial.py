import numpy as np

from src.spatial import (
    analytical_coverage_prob,
    compute_distances,
    empirical_coverage_prob,
    generate_positions,
)


class TestPositionGeneration:
    def test_shape(self) -> None:
        pos = generate_positions(100, 50.0, seed=42)
        assert pos.shape == (100, 2)

    def test_bounds(self) -> None:
        pos = generate_positions(1000, 50.0, seed=42)
        assert np.all(pos >= 0)
        assert np.all(pos <= 50.0)

    def test_reproducibility(self) -> None:
        pos1 = generate_positions(10, 50.0, seed=42)
        pos2 = generate_positions(10, 50.0, seed=42)
        np.testing.assert_array_equal(pos1, pos2)


class TestCoverage:
    def test_zero_volunteers(self) -> None:
        assert analytical_coverage_prob(0, 10.0, 100.0) == 0.0

    def test_monotonicity(self) -> None:
        probs = [analytical_coverage_prob(k, 10.0, 100.0) for k in range(101)]
        assert all(probs[i] <= probs[i + 1] for i in range(100))

    def test_limit(self) -> None:
        prob = analytical_coverage_prob(10000, 10.0, 100.0)
        assert prob > 0.999

    def test_empirical_matches_analytical(self) -> None:
        k, R, L = 50, 10.0, 100.0
        analytical = analytical_coverage_prob(k, R, L)
        empirical, std_err = empirical_coverage_prob(k, R, L, n_samples=50000, seed=42)
        assert abs(empirical - analytical) < 3 * std_err


class TestDistances:
    def test_zero_distance(self) -> None:
        pos = np.array([[5.0, 5.0]])
        target = np.array([5.0, 5.0])
        dist = compute_distances(pos, target)
        assert dist[0] == 0.0

    def test_known_distance(self) -> None:
        pos = np.array([[0.0, 0.0]])
        target = np.array([3.0, 4.0])
        dist = compute_distances(pos, target)
        assert dist[0] == 5.0
