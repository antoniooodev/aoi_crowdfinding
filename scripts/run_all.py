"""
Run all experiments and generate all required outputs.
Usage (from repo root): python -m experiments.run_all
"""

from __future__ import annotations

from experiments.exp01_equilibrium_analysis import run as run_exp01
from experiments.exp02_social_optimum import run as run_exp02
from experiments.exp03_price_of_anarchy import run as run_exp03
from experiments.exp04_stackelberg import run as run_exp04
from experiments.exp05_sensitivity import run as run_exp05
from experiments.exp06_heterogeneous_analysis import run as run_exp06


def generate_figures() -> None:
    run_exp01()
    run_exp02()
    run_exp03()
    run_exp04()
    run_exp05()
    run_exp06()


def main() -> None:
    generate_figures()


if __name__ == "__main__":
    main()