"""
AoI Crowd-Finding: Game-Theoretic Analysis of Age of Information
in Emergency Search Networks

This package implements the theoretical model and simulation for analyzing
strategic volunteer participation in crowd-finding networks using Age of
Information as the performance metric.

Modules:
    config - Configuration and parameter management
    spatial - 2D geometry and coverage calculations
    aoi - Age of Information computations
    game - Game theory (Nash equilibrium, social optimum, PoA)
    simulation - Monte Carlo simulation engine
    stackelberg - Stackelberg game with platform incentives
    visualization - Publication-quality plotting

CORRECTIONS APPLIED (v1.1):
    1. Target sampled from interior [R, L-R]² to match analytical theory
    2. Cost ranges computed adaptively based on thresholds (Bρ, NBρ)
    3. PoA heatmap masks infinite values (where k*=0)
    4. Stackelberg incentive uses epsilon for robust discrete equilibrium
"""

from .config import (
    SimConfig,
    PhysicalParams,
    GameParams,
    SimulationParams,
    ExperimentGrid,
    DEFAULT_CONFIG,
    EXPERIMENT_GRID,
    compute_informative_cost_range,
    compute_normalized_cost_range,
    print_thresholds,
    suggest_parameters,
)

from .spatial import (
    generate_positions,
    generate_target,
    compute_distances,
    coverage_mask,
    count_covering,
    analytical_coverage_prob,
    empirical_coverage_prob,
    coverage_probability_vectorized,
    compute_rho,
    compute_thresholds,
)

from .aoi import (
    expected_aoi,
    expected_aoi_from_k,
    aoi_vectorized,
    simulate_aoi_trajectory,
    time_average_aoi,
    peak_aoi,
    marginal_aoi_reduction,
)

from .game import (
    utility_active,
    utility_inactive,
    marginal_utility,
    find_nash_equilibrium,
    find_nash_equilibrium_search,
    social_welfare,
    find_social_optimum,
    find_social_optimum_search,
    price_of_anarchy,
    participation_gap,
    analyze_equilibrium,
    sweep_cost,
)

from .simulation import (
    Simulation,
    MonteCarloSimulation,
    SimulationResult,
    MonteCarloResult,
    run_parameter_sweep,
    validate_analytical,
)

from .stackelberg import (
    induced_equilibrium,
    optimal_incentive,
    optimal_incentive_for_target,
    robust_optimal_incentive,
    total_incentive_cost,
    platform_objective,
    analyze_stackelberg,
    incentive_sensitivity,
)

__version__ = '1.1.0'
__author__ = 'Antonio'
