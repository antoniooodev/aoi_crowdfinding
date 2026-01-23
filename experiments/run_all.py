"""Run the full experiment suite.

Usage (from repo root):
    python -m experiments.run_all

This script executes each experiment module and writes results under `results/` (CSV data and figures).
"""


from .exp01_equilibrium_analysis import run as run_exp01
from .exp02_social_optimum import run as run_exp02
from .exp03_price_of_anarchy import run as run_exp03
from .exp04_stackelberg import run as run_exp04
from .exp05_sensitivity import run as run_exp05
from .exp06_heterogeneous_analysis import run as run_exp06


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