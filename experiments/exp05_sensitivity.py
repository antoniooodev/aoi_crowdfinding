"""Experiment 05: Analytical validation and spatial snapshot.

Validates analytical AoI against Monte Carlo simulation for selected k values and produces a single spatial snapshot illustrating one realization.

Outputs:
    results/data/validation_results.csv
    results/figures/spatial_snapshot.{pdf,png}

Run:
    python -m experiments.exp05_sensitivity
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PhysicalParams, SimulationParams, SimConfig, GameParams
from src.simulation import validate_analytical
from src.spatial import generate_positions
from src.game import find_nash_equilibrium
from src.visualization import plot_spatial_snapshot

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "data"


def run() -> pd.DataFrame:
    L = 100.0
    R = 10.0
    N = 100
    B = 10.0
    c = 1.0

    cfg = SimConfig(
        physical=PhysicalParams(L=L, R=R),
        game=GameParams(N=N, B=B, c=c),
        simulation=SimulationParams(T=8000, n_runs=1, seed=42),
    )

    k_values = [5, 10, 20, 50, 80]
    out = validate_analytical(cfg, k_values=k_values, n_runs=300, tolerance=0.05)
    
    
    df = pd.DataFrame(
        {
            "k": out["k"].astype(int),
            "analytical_aoi": out["analytical_aoi"].astype(float),
            "simulated_aoi": out["simulated_aoi"].astype(float),
            "simulated_aoi_std": out["simulated_aoi_std"].astype(float),
            "relative_error": out["relative_error"].astype(float),
            "passed": out["within_tolerance"].astype(bool),
            "within_3sigma": out["within_3sigma"].astype(bool),
            "L": float(L),
            "R": float(R),
            "N": int(N),
            "B": float(B),
            "c": float(c),
        }
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "validation_results.csv", index=False)

    rng = np.random.default_rng(7)
    positions = generate_positions(N, L, seed=7)
    target = rng.uniform(R, L - R, size=(2,))
    k_star = int(find_nash_equilibrium(N, R, L, B, c))

    active_mask = np.zeros(N, dtype=bool)
    if k_star > 0:
        active_idx = rng.choice(N, size=k_star, replace=False)
        active_mask[active_idx] = True

    plot_spatial_snapshot(
        positions=positions.astype(float),
        target=target.astype(float),
        active_mask=active_mask,
        R=float(R),
        L=float(L),
    )

    return df


def main() -> None:
    run()


if __name__ == "__main__":
    main()
