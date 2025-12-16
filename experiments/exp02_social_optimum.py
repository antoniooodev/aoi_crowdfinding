"""
Experiment (AoI vs k): analytical vs simulated.
Generates: results/figures/aoi_vs_k.{pdf,png}
"""

from __future__ import annotations

import numpy as np

from src.aoi import expected_aoi_from_k
from src.config import PhysicalParams, SimulationParams, SimConfig, GameParams
from src.simulation import run_parameter_sweep
from src.visualization import plot_aoi_vs_k


def run() -> None:
    L = 100.0
    R = 10.0
    N = 100
    B = 10.0
    c = 1.0

    k_values = np.arange(0, N + 1, dtype=int)

    cfg = SimConfig(
        physical=PhysicalParams(L=L, R=R),
        game=GameParams(N=N, B=B, c=c),
        simulation=SimulationParams(T=5000, n_runs=1, seed=123),
    )

    aoi_analytical = np.array([expected_aoi_from_k(int(k), R, L) for k in k_values], dtype=float)

    sweep = run_parameter_sweep(
        base_config=cfg,
        k_values=k_values.tolist(),
        n_runs=200,
        show_progress=True,
    )
    aoi_sim = sweep["mean_aoi"].astype(float)
    aoi_std = sweep["mean_aoi_std"].astype(float)

    plot_aoi_vs_k(
        k_values=k_values,
        aoi_analytical=aoi_analytical,
        aoi_simulated=aoi_sim,
        aoi_std=aoi_std,
        params={"N": N, "R": float(R), "L": float(L)},
    )



def main() -> None:
    run()


if __name__ == "__main__":
    main()
