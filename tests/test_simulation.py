"""Unit tests for `src.simulation`.

Validates single-run and Monte Carlo outputs (shapes, basic invariants) and cross-checks summary statistics.
"""

import numpy as np

from src.config import PhysicalParams, SimulationParams, SimConfig
from src.simulation import MonteCarloSimulation, Simulation, validate_analytical


class TestSimulation:
    def test_step_zero_volunteers(self) -> None:
        cfg = SimConfig(
            physical=PhysicalParams(L=100.0, R=10.0),
            simulation=SimulationParams(T=3, n_runs=1, seed=123),
        )
        sim = Simulation(cfg)

        info1 = sim.step(0)
        assert info1["detected"] is False
        assert info1["aoi"] == 1
        assert info1["t"] == 1

        info2 = sim.step(0)
        assert info2["detected"] is False
        assert info2["aoi"] == 2
        assert info2["t"] == 2

    def test_run_shape_and_consistency(self) -> None:
        cfg = SimConfig(
            physical=PhysicalParams(L=100.0, R=10.0),
            simulation=SimulationParams(T=200, n_runs=1, seed=7),
        )
        sim = Simulation(cfg)
        res = sim.run(k=5, store_trajectory=True)

        assert res.trajectory is not None
        assert res.trajectory.shape == (cfg.simulation.T,)
        assert np.all(res.trajectory >= 0)
        assert abs(res.mean_aoi - float(np.mean(res.trajectory))) < 1e-12
        assert res.peak_aoi == int(np.max(res.trajectory))


class TestMonteCarlo:
    def test_reproducible(self) -> None:
        cfg = SimConfig(
            physical=PhysicalParams(L=100.0, R=10.0),
            simulation=SimulationParams(T=500, n_runs=10, seed=11),
        )
        mc1 = MonteCarloSimulation(cfg)
        mc2 = MonteCarloSimulation(cfg)

        r1 = mc1.run(k=10, n_runs=10, show_progress=False, store_individual=True)
        r2 = mc2.run(k=10, n_runs=10, show_progress=False, store_individual=True)

        assert r1.mean_aoi == r2.mean_aoi
        assert r1.mean_aoi_std == r2.mean_aoi_std
        assert r1.mean_peak_aoi == r2.mean_peak_aoi
        assert r1.mean_detection_rate == r2.mean_detection_rate
        assert r1.n_runs == r2.n_runs == 10


class TestValidation:
    def test_validate_analytical_smoke(self) -> None:
        cfg = SimConfig(
            physical=PhysicalParams(L=200.0, R=10.0),
            simulation=SimulationParams(T=5000, n_runs=1, seed=42),
        )
        out = validate_analytical(cfg, k_values=[10, 20, 50], n_runs=20, tolerance=0.2)
        assert isinstance(out["all_passed"], bool)
        assert out["k"].shape[0] == 3
