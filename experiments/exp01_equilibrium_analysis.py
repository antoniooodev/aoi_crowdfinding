"""Experiment 01: Equilibrium and social optimum vs cost.

Sweeps cost `c` over a parameter grid and compares the Nash equilibrium participation k* against the socially optimal participation k_opt.

Outputs:
    results/data/equilibrium_analysis.csv
    results/figures/equilibrium_vs_cost.{pdf,png}

Run:
    python -m experiments.exp01_equilibrium_analysis
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.game import find_nash_equilibrium, find_social_optimum
from src.visualization import plot_equilibrium_vs_cost

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "data"


def run() -> pd.DataFrame:
    N_values = [50, 100, 200, 500]
    c_values = np.linspace(0.1, 10.0, 50, dtype=float)
    R_L_ratios = [0.05, 0.10, 0.15, 0.20]
    B = 10.0
    L = 100.0

    rows: list[dict] = []
    for N in N_values:
        for ratio in R_L_ratios:
            R = float(ratio * L)
            rho = float(np.pi * R * R / (L * L))
            for c in c_values:
                k_star = int(find_nash_equilibrium(N, R, L, B, float(c)))
                k_opt = int(find_social_optimum(N, R, L, B, float(c)))
                gap_approx = float(np.log(N) / rho) if rho > 0 else float("inf")
                rows.append(
                    {
                        "N": int(N),
                        "L": float(L),
                        "R": float(R),
                        "R_over_L": float(ratio),
                        "rho": float(rho),
                        "B": float(B),
                        "c": float(c),
                        "k_star": k_star,
                        "k_opt": k_opt,
                        "gap": int(k_opt - k_star),
                        "gap_approx": float(gap_approx),
                    }
                )

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "equilibrium_analysis.csv", index=False)

    baseline_N = 100
    baseline_ratio = 0.10
    sub = df[(df["N"] == baseline_N) & (np.isclose(df["R_over_L"], baseline_ratio))].copy()
    sub.sort_values("c", inplace=True)

    R_plot = float(baseline_ratio * L)

    plot_equilibrium_vs_cost(
        c_values=sub["c"].to_numpy(dtype=float),
        k_ne=sub["k_star"].to_numpy(dtype=int),
        k_opt=sub["k_opt"].to_numpy(dtype=int),
        params={"N": baseline_N, "R": R_plot, "L": float(L)},
    )

    return df


def main() -> None:
    run()


if __name__ == "__main__":
    main()
