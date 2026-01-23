"""Experiment 04: Stackelberg incentive design.

For each (N, R/L, c) setting, computes an incentive p* that induces a participation level close to the social optimum and compares welfare across regimes.

Outputs:
    results/data/stackelberg_analysis.csv
    results/figures/stackelberg_incentive.{pdf,png}
    results/figures/welfare_comparison.{pdf,png}

Run:
    python -m experiments.exp04_stackelberg
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.game import find_nash_equilibrium, find_social_optimum, social_welfare
from src.stackelberg import induced_equilibrium, optimal_incentive, total_incentive_cost
from src.visualization import plot_stackelberg_incentive, plot_welfare_comparison

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
            for c in c_values:
                k_star = int(find_nash_equilibrium(N, R, L, B, float(c)))
                k_opt = int(find_social_optimum(N, R, L, B, float(c)))
                p_star = float(optimal_incentive(N, R, L, B, float(c)))
                k_induced = int(induced_equilibrium(p_star, N, R, L, B, float(c)))
                total_cost = float(total_incentive_cost(p_star, N, R, L, B, float(c)))

                W_star = float(social_welfare(k_star, N, R, L, B, float(c)))
                W_opt = float(social_welfare(k_opt, N, R, L, B, float(c)))
                W_stack = float(social_welfare(k_induced, N, R, L, B, float(c)))

                rows.append(
                    {
                        "N": int(N),
                        "L": float(L),
                        "R": float(R),
                        "R_over_L": float(ratio),
                        "B": float(B),
                        "c": float(c),
                        "k_star": int(k_star),
                        "k_opt": int(k_opt),
                        "p_star": float(p_star),
                        "k_induced": int(k_induced),
                        "total_incentive_cost": float(total_cost),
                        "W_star": float(W_star),
                        "W_opt": float(W_opt),
                        "W_stackelberg": float(W_stack),
                    }
                )

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "stackelberg_analysis.csv", index=False)

    baseline_N = 100
    baseline_ratio = 0.10
    sub = df[(df["N"] == baseline_N) & (np.isclose(df["R_over_L"], baseline_ratio))].copy()
    sub.sort_values("c", inplace=True)

    plot_stackelberg_incentive(
    c_values=sub["c"].to_numpy(dtype=float),
    p_star=sub["p_star"].to_numpy(dtype=float),
    total_incentive=sub["total_incentive_cost"].to_numpy(dtype=float),
    params={"N": baseline_N, "R": float(baseline_ratio * L), "L": float(L)},
    )

    plot_welfare_comparison(
        c_values=sub["c"].to_numpy(dtype=float),
        W_ne=sub["W_star"].to_numpy(dtype=float),
        W_opt=sub["W_opt"].to_numpy(dtype=float),
        W_stack=sub["W_stackelberg"].to_numpy(dtype=float),
        params={"N": baseline_N, "R": float(baseline_ratio * L), "L": float(L)},
    )

    return df


def main() -> None:
    run()


if __name__ == "__main__":
    main()
