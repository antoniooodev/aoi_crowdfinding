"""AoI Crowd-Finding package.

Core library for the game-theoretic crowd-finding model and simulations, using Age of Information as the performance metric.

Public modules: config, spatial, aoi, game, simulation, stackelberg, visualization.

Notes:
    v1.1 fixes: target is sampled from the interior [R, L-R]^2; cost ranges are derived from model thresholds; PoA plots mask infinite values (k*=0); Stackelberg incentives use a small epsilon for discrete tie-breaking.
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
