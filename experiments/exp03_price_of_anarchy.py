"""
Experiment 2: Price of Anarchy heatmap over (N, c).
Saves: results/data/poa_grid.csv
Generates: results/figures/poa_heatmap.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.game import (
    find_nash_equilibrium,
    find_social_optimum,
    price_of_anarchy,
    social_welfare,
)
from src.visualization import plot_poa_heatmap

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "data"


def run() -> pd.DataFrame:
    N_values = [50, 100, 200, 500]
    c_values = np.linspace(0.1, 10.0, 50, dtype=float)
    B = 10.0
    L = 100.0
    R = 10.0

    poa_matrix = np.zeros((len(N_values), len(c_values)), dtype=float)
    rows: list[dict] = []

    for i, N in enumerate(N_values):
        for j, c in enumerate(c_values):
            k_star = int(find_nash_equilibrium(N, R, L, B, float(c)))
            k_opt = int(find_social_optimum(N, R, L, B, float(c)))
            W_star = float(social_welfare(k_star, N, R, L, B, float(c)))
            W_opt = float(social_welfare(k_opt, N, R, L, B, float(c)))
            poa = float(price_of_anarchy(N, R, L, B, float(c)))

            poa_matrix[i, j] = poa
            rows.append(
                {
                    "N": int(N),
                    "c": float(c),
                    "L": float(L),
                    "R": float(R),
                    "B": float(B),
                    "k_star": int(k_star),
                    "k_opt": int(k_opt),
                    "W_star": float(W_star),
                    "W_opt": float(W_opt),
                    "poa": float(poa),
                }
            )

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "poa_grid.csv", index=False)

    poa_plot = poa_matrix.copy()
    finite = np.isfinite(poa_plot)
    if np.any(finite):
        vmax = float(np.nanpercentile(poa_plot[finite], 95))
        poa_plot[~finite] = vmax

    plot_poa_heatmap(
        N_values=np.array(N_values, dtype=int),
        c_values=c_values.astype(float),
        poa_matrix=poa_plot.astype(float),
    )

    return df


def main() -> None:
    run()


if __name__ == "__main__":
    main()
