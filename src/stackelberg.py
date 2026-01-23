"""Stackelberg incentive design utilities.

Computes incentives that induce desired equilibrium participation and compares welfare under platform payments.

Notes:
    A small epsilon is used for deterministic tie-breaking when agents are indifferent in the discrete equilibrium.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .spatial import analytical_coverage_prob, compute_rho
from .game import find_nash_equilibrium, social_welfare, find_social_optimum


# Small epsilon for tie-breaking (when agent is indifferent, choose to participate)
EPSILON = 1e-9


def induced_equilibrium(
    p: float,
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> int:
    """
    Compute NE induced by incentive p.
    
    With incentive p, effective cost is (c - p).
    
    Parameters
    ----------
    p : float
        Per-volunteer incentive payment
    N, R, L, B, c : parameters
    
    Returns
    -------
    k : int
        Induced equilibrium participation
    """
    effective_cost = max(0, c - p)
    return find_nash_equilibrium(N, R, L, B, effective_cost)


def optimal_incentive_for_target(
    k_target: int,
    R: float,
    L: float,
    B: float,
    c: float,
    add_epsilon: bool = False
) -> float:
    """
    Compute incentive needed to induce target participation k_target.
    
    p* = c - B * ρ * (1 - ρ)^(k_target - 1) + ε
    
    Notes:
        When add_epsilon=True, a small epsilon is added for deterministic
        tie-breaking in the discrete equilibrium (indifference cases).
    
    Parameters
    ----------
    k_target : int
        Target number of active volunteers
    R, L, B, c : parameters
    add_epsilon : bool
        If True, add small epsilon for robustness
    
    Returns
    -------
    p : float
        Required incentive (clamped to [0, c])
    """
    if k_target <= 0:
        return 0.0
    
    rho = compute_rho(R, L)
    
    # Threshold cost at which k_target-th volunteer is indifferent
    threshold = B * rho * (1 - rho) ** (k_target - 1)
    
    # Incentive to make participation worthwhile
    p_star = c - threshold
    
    # Add epsilon to push past indifference
    if add_epsilon:
        p_star += EPSILON
    
    # Match the closed-form incentive used in the analytical tests.
    return max(0.0, p_star)


def optimal_incentive(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float,
    add_epsilon: bool = False
) -> float:
    """
    Compute optimal incentive to implement social optimum.
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    add_epsilon : bool
        If True, add small epsilon for robustness
    
    Returns
    -------
    p_star : float
        Optimal incentive
    """
    k_opt = find_social_optimum(N, R, L, B, c)
    return optimal_incentive_for_target(k_opt, R, L, B, c, add_epsilon)


def robust_optimal_incentive(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float,
    resolution: int = 1000
) -> Tuple[float, int]:
    """
    Find optimal incentive by direct verification.

    This searches for the minimum p such that induced_equilibrium(p) >= k_opt,
    which is more robust to discrete tie cases than relying on a closed-form
    expression alone.
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    resolution : int
        Grid resolution for search
    
    Returns
    -------
    p_star : float
        Optimal incentive
    k_induced : int
        Induced equilibrium participation
    """
    k_opt = find_social_optimum(N, R, L, B, c)
    
    if k_opt == 0:
        return 0.0, 0
    
    # Binary search for minimum p that achieves k >= k_opt
    p_low, p_high = 0.0, c
    
    # First check if any incentive can achieve k_opt
    k_max = induced_equilibrium(c, N, R, L, B, c)
    if k_max < k_opt:
        # Even with full subsidy, can't reach k_opt
        return c, k_max
    
    # Binary search
    for _ in range(50):  # ~15 decimal places precision
        p_mid = (p_low + p_high) / 2
        k_mid = induced_equilibrium(p_mid, N, R, L, B, c)
        
        if k_mid >= k_opt:
            p_high = p_mid
        else:
            p_low = p_mid
        
        if p_high - p_low < 1e-12:
            break
    
    # Use slightly higher than minimum to ensure k >= k_opt
    p_star = p_high + EPSILON
    p_star = min(c, p_star)
    k_induced = induced_equilibrium(p_star, N, R, L, B, c)
    
    return p_star, k_induced


def total_incentive_cost(
    p: float,
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> float:
    """
    Compute total incentive payment by platform.
    
    Total = p * k(p)
    
    Parameters
    ----------
    p : float
        Per-volunteer incentive
    N, R, L, B, c : parameters
    
    Returns
    -------
    total : float
        Total platform expenditure
    """
    k = induced_equilibrium(p, N, R, L, B, c)
    return p * k


def platform_objective(
    p: float,
    N: int,
    R: float,
    L: float,
    B: float,
    c: float,
    budget: Optional[float] = None
) -> float:
    """
    Compute platform's objective (social welfare, since transfers cancel).
    
    V(p) = W(k(p))
    
    Note: Incentive payments are transfers, so social welfare equals
    N*B*P_det(k) - k*c regardless of p (the p*k terms cancel between
    platform and volunteers).
    
    Parameters
    ----------
    p : float
        Per-volunteer incentive
    N, R, L, B, c : parameters
    budget : float, optional
        Maximum total incentive budget
    
    Returns
    -------
    objective : float
        Platform objective value (-inf if budget exceeded)
    """
    k = induced_equilibrium(p, N, R, L, B, c)
    total_payment = p * k
    
    if budget is not None and total_payment > budget:
        return -np.inf
    
    return social_welfare(k, N, R, L, B, c)


def analyze_stackelberg(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float,
    budget: Optional[float] = None,
    use_robust: bool = True
) -> Dict:
    """
    Complete Stackelberg analysis.
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    budget : float, optional
        Maximum total incentive
    use_robust : bool
        If True, use robust incentive search (recommended)
    
    Returns
    -------
    results : dict
        Comprehensive analysis results
    """
    rho = compute_rho(R, L)
    
    # No-incentive baseline
    k_ne = find_nash_equilibrium(N, R, L, B, c)
    W_ne = social_welfare(k_ne, N, R, L, B, c)
    
    # Social optimum (unconstrained)
    k_opt = find_social_optimum(N, R, L, B, c)
    W_opt = social_welfare(k_opt, N, R, L, B, c)
    
    # Optimal incentive
    if use_robust:
        p_opt, k_stack = robust_optimal_incentive(N, R, L, B, c)
    else:
        p_opt = optimal_incentive(N, R, L, B, c, add_epsilon=True)
        k_stack = induced_equilibrium(p_opt, N, R, L, B, c)
    
    total_payment_opt = p_opt * k_stack
    
    # Check if budget allows social optimum
    budget_sufficient = budget is None or total_payment_opt <= budget
    
    if not budget_sufficient:
        # Find best within budget via search
        p_stack, k_stack, W_stack = find_optimal_incentive_search(
            N, R, L, B, c, budget=budget
        )
    else:
        W_stack = social_welfare(k_stack, N, R, L, B, c)
        p_stack = p_opt
    
    # Compute efficiency metrics
    if W_ne > 0:
        efficiency_gain = (W_stack - W_ne) / W_ne
    else:
        efficiency_gain = np.inf if W_stack > 0 else 0.0
    
    if W_opt > 0:
        optimality_gap = (W_opt - W_stack) / W_opt
    else:
        optimality_gap = 0.0
    
    return {
        'N': N,
        'R': R,
        'L': L,
        'B': B,
        'c': c,
        'rho': rho,
        'budget': budget,
        # No incentive
        'k_ne': k_ne,
        'welfare_ne': W_ne,
        # Social optimum
        'k_opt': k_opt,
        'welfare_opt': W_opt,
        'p_for_optimum': p_opt,
        'cost_for_optimum': total_payment_opt,
        # Stackelberg solution
        'k_stackelberg': k_stack,
        'p_stackelberg': p_stack,
        'welfare_stackelberg': W_stack,
        'incentive_cost': p_stack * k_stack,
        # Metrics
        'budget_sufficient': budget_sufficient,
        'efficiency_gain': efficiency_gain,
        'optimality_gap': optimality_gap,
        'k_matches_opt': k_stack == k_opt,
    }


def find_optimal_incentive_search(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float,
    budget: Optional[float] = None,
    resolution: int = 100
) -> Tuple[float, int, float]:
    """
    Find optimal incentive by grid search (for budget-constrained case).
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    budget : float, optional
        Maximum total incentive
    resolution : int
        Grid resolution
    
    Returns
    -------
    p_star : float
        Optimal incentive
    k_star : int
        Induced participation
    welfare : float
        Resulting social welfare
    """
    p_values = np.linspace(0, c, resolution)
    
    best_p = 0.0
    best_k = 0
    best_welfare = -np.inf
    
    for p in p_values:
        k = induced_equilibrium(p, N, R, L, B, c)
        
        # Check budget constraint
        if budget is not None and p * k > budget:
            continue
        
        welfare = social_welfare(k, N, R, L, B, c)
        
        if welfare > best_welfare:
            best_welfare = welfare
            best_p = p
            best_k = k
    
    return best_p, best_k, best_welfare


def incentive_sensitivity(
    N: int,
    R: float,
    L: float,
    B: float,
    c_values: np.ndarray,
    use_robust: bool = True
) -> Dict[str, np.ndarray]:
    """
    Analyze how optimal incentive varies with cost.
    
    Parameters
    ----------
    N, R, L, B : parameters
    c_values : array
        Cost values to analyze
    use_robust : bool
        Use robust incentive computation
    
    Returns
    -------
    results : dict of arrays
        Arrays of results for each cost value
    """
    n = len(c_values)
    results = {
        'c': c_values.copy(),
        'k_ne': np.zeros(n, dtype=int),
        'k_opt': np.zeros(n, dtype=int),
        'k_stackelberg': np.zeros(n, dtype=int),
        'p_star': np.zeros(n),
        'total_incentive': np.zeros(n),
        'welfare_ne': np.zeros(n),
        'welfare_opt': np.zeros(n),
        'welfare_stackelberg': np.zeros(n),
        'poa': np.zeros(n),
        'k_matches_opt': np.zeros(n, dtype=bool),
    }
    
    for i, c in enumerate(c_values):
        analysis = analyze_stackelberg(N, R, L, B, c, use_robust=use_robust)
        
        results['k_ne'][i] = analysis['k_ne']
        results['k_opt'][i] = analysis['k_opt']
        results['k_stackelberg'][i] = analysis['k_stackelberg']
        results['p_star'][i] = analysis['p_stackelberg']
        results['total_incentive'][i] = analysis['incentive_cost']
        results['welfare_ne'][i] = analysis['welfare_ne']
        results['welfare_opt'][i] = analysis['welfare_opt']
        results['welfare_stackelberg'][i] = analysis['welfare_stackelberg']
        results['k_matches_opt'][i] = analysis['k_matches_opt']
        
        if results['welfare_ne'][i] > 0:
            results['poa'][i] = results['welfare_opt'][i] / results['welfare_ne'][i]
        else:
            results['poa'][i] = np.inf
    
    return results


# =============================================================================
# HETEROGENEOUS COST STACKELBERG (Version 2.0)
# =============================================================================

from .config import HeterogeneousCostParams
from .game import (
    find_nash_heterogeneous,
    find_social_optimum_heterogeneous_expected,
    expected_welfare_heterogeneous,
    nash_threshold
)


def induced_equilibrium_heterogeneous(
    p: float,
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams,
    tol: float = 1e-8,
    max_iter: int = 100
) -> Tuple[int, float]:
    """
    Compute equilibrium induced by incentive p with heterogeneous costs.
    
    With incentive p, a volunteer participates if c_i <= c̄(k) + p.
    Fixed-point: k = N · F(Bρ(1-ρ)^k + p)
    
    Parameters
    ----------
    p : float
        Per-volunteer incentive
    N : int
        Number of volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    cost_params : HeterogeneousCostParams
        Cost distribution parameters
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    k_induced : int
        Induced participation
    c_bar_induced : float
        Induced threshold (including incentive)
    """
    k = N / 2  # Initial guess
    
    for _ in range(max_iter):
        c_bar = nash_threshold(int(round(k)), rho, B) + p
        c_bar = np.clip(c_bar, cost_params.c_min, cost_params.c_max)
        k_new = N * cost_params.cdf(c_bar)
        
        if abs(k_new - k) < tol:
            break
        k = k_new
    
    k_star = max(0, min(N, int(round(k))))
    c_bar_star = nash_threshold(k_star, rho, B) + p
    
    return k_star, c_bar_star


def optimal_incentive_heterogeneous(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams
) -> Tuple[float, int, float]:
    """
    Compute optimal incentive for heterogeneous costs.
    
    p* = c̄_opt - Bρ(1-ρ)^k_opt
    
    where c̄_opt is the threshold that selects k_opt lowest-cost volunteers.
    
    Parameters
    ----------
    N : int
        Number of volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    cost_params : HeterogeneousCostParams
        Cost distribution parameters
    
    Returns
    -------
    p_star : float
        Optimal per-volunteer incentive
    k_induced : int
        Participation induced by p*
    c_bar_opt : float
        Optimal cost threshold
    """
    k_opt, c_bar_opt = find_social_optimum_heterogeneous_expected(N, rho, B, cost_params)
    
    if k_opt == 0:
        return 0.0, 0, cost_params.c_min
    
    # Incentive to shift threshold from c̄* to c̄_opt
    c_bar_nash = nash_threshold(k_opt, rho, B)
    p_star = max(0.0, c_bar_opt - c_bar_nash + EPSILON)
    
    # Verify induced equilibrium
    k_induced, _ = induced_equilibrium_heterogeneous(p_star, N, rho, B, cost_params)
    
    return p_star, k_induced, c_bar_opt


def total_incentive_cost_heterogeneous(
    p: float,
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams
) -> float:
    """
    Compute total incentive payment for heterogeneous case.
    
    Parameters
    ----------
    p : float
        Per-volunteer incentive
    N, rho, B : parameters
    cost_params : HeterogeneousCostParams
    
    Returns
    -------
    total : float
        Total incentive payment p * k(p)
    """
    k, _ = induced_equilibrium_heterogeneous(p, N, rho, B, cost_params)
    return p * k


def expected_total_budget_heterogeneous(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams
) -> float:
    """
    Expected total budget needed to achieve social optimum.
    
    I* = k_opt · (c̄_opt - c̄*)
    
    Parameters
    ----------
    N, rho, B : parameters
    cost_params : HeterogeneousCostParams
    
    Returns
    -------
    budget : float
        Expected total incentive budget
    """
    k_star, c_bar_star = find_nash_heterogeneous(N, rho, B, cost_params)
    k_opt, c_bar_opt = find_social_optimum_heterogeneous_expected(N, rho, B, cost_params)
    
    if k_opt <= k_star:
        return 0.0  # No incentive needed
    
    p_star, _, _ = optimal_incentive_heterogeneous(N, rho, B, cost_params)
    return p_star * k_opt


def find_optimal_incentive_heterogeneous_search(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams,
    budget: float,
    resolution: int = 100
) -> Tuple[float, int]:
    """
    Find best incentive within budget via grid search.
    
    Parameters
    ----------
    N, rho, B : parameters
    cost_params : HeterogeneousCostParams
    budget : float
        Maximum total budget
    resolution : int
        Grid resolution
    
    Returns
    -------
    p_star : float
        Best incentive within budget
    k_star : int
        Induced participation
    """
    p_max = cost_params.c_max  # Maximum sensible incentive
    best_p, best_k, best_welfare = 0.0, 0, -np.inf
    
    for p in np.linspace(0, p_max, resolution):
        k, _ = induced_equilibrium_heterogeneous(p, N, rho, B, cost_params)
        if p * k > budget:
            continue
        welfare = expected_welfare_heterogeneous(k, N, rho, B, cost_params)
        if welfare > best_welfare:
            best_welfare, best_p, best_k = welfare, p, k
    
    return best_p, best_k


def analyze_stackelberg_heterogeneous(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams,
    budget: Optional[float] = None
) -> Dict:
    """
    Complete Stackelberg analysis with heterogeneous costs.
    
    Parameters
    ----------
    N : int
        Number of volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    cost_params : HeterogeneousCostParams
        Cost distribution parameters
    budget : float, optional
        Maximum total incentive budget
    
    Returns
    -------
    results : dict
        Comprehensive analysis results
    """
    # Nash equilibrium (no incentive)
    k_ne, c_bar_ne = find_nash_heterogeneous(N, rho, B, cost_params)
    W_ne = expected_welfare_heterogeneous(k_ne, N, rho, B, cost_params)
    
    # Social optimum
    k_opt, c_bar_opt = find_social_optimum_heterogeneous_expected(N, rho, B, cost_params)
    W_opt = expected_welfare_heterogeneous(k_opt, N, rho, B, cost_params)
    
    # Optimal incentive
    p_opt, k_induced, _ = optimal_incentive_heterogeneous(N, rho, B, cost_params)
    total_payment = p_opt * k_induced
    
    budget_sufficient = budget is None or total_payment <= budget
    
    if not budget_sufficient:
        # Find best incentive within budget
        p_stack, k_stack = find_optimal_incentive_heterogeneous_search(N, rho, B, cost_params, budget)
    else:
        p_stack, k_stack = p_opt, k_induced
    
    W_stack = expected_welfare_heterogeneous(k_stack, N, rho, B, cost_params)
    
    # Compute efficiency metrics
    if W_ne > 0:
        efficiency_gain = (W_stack - W_ne) / W_ne
    else:
        efficiency_gain = np.inf if W_stack > 0 else 0.0
    
    if W_opt > 0:
        optimality_gap = (W_opt - W_stack) / W_opt
    else:
        optimality_gap = 0.0
    
    return {
        'N': N,
        'rho': rho,
        'B': B,
        'c_min': cost_params.c_min,
        'c_max': cost_params.c_max,
        'mean_cost': cost_params.mean_cost,
        'distribution': cost_params.distribution,
        'budget': budget,
        # No incentive
        'k_ne': k_ne,
        'c_bar_ne': c_bar_ne,
        'welfare_ne': W_ne,
        # Social optimum
        'k_opt': k_opt,
        'c_bar_opt': c_bar_opt,
        'welfare_opt': W_opt,
        # Incentive to achieve optimum
        'p_for_optimum': p_opt,
        'cost_for_optimum': total_payment,
        # Stackelberg solution (possibly budget-constrained)
        'k_stackelberg': k_stack,
        'p_stackelberg': p_stack,
        'welfare_stackelberg': W_stack,
        'incentive_cost': p_stack * k_stack,
        # Metrics
        'budget_sufficient': budget_sufficient,
        'efficiency_gain': efficiency_gain,
        'optimality_gap': optimality_gap,
        'k_matches_opt': k_stack >= k_opt,
        'participation_gap_closed': k_opt - k_ne if k_ne < k_opt else 0,
    }


def incentive_sensitivity_heterogeneous(
    N: int,
    rho: float,
    B: float,
    c_min_values: np.ndarray,
    c_max_values: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Analyze how optimal incentive varies with cost distribution.
    
    Parameters
    ----------
    N, rho, B : parameters
    c_min_values : array
        Minimum costs to test
    c_max_values : array
        Maximum costs to test (same length as c_min_values)
    
    Returns
    -------
    results : dict of arrays
    """
    n = len(c_min_values)
    results = {
        'c_min': c_min_values.copy(),
        'c_max': c_max_values.copy(),
        'mean_cost': np.zeros(n),
        'k_ne': np.zeros(n, dtype=int),
        'k_opt': np.zeros(n, dtype=int),
        'k_stackelberg': np.zeros(n, dtype=int),
        'p_star': np.zeros(n),
        'total_incentive': np.zeros(n),
        'welfare_ne': np.zeros(n),
        'welfare_opt': np.zeros(n),
        'welfare_stackelberg': np.zeros(n),
        'poa': np.zeros(n),
        'efficiency_gain': np.zeros(n),
    }
    
    for i in range(n):
        cost_params = HeterogeneousCostParams(c_min=c_min_values[i], c_max=c_max_values[i])
        a = analyze_stackelberg_heterogeneous(N, rho, B, cost_params)
        
        results['mean_cost'][i] = cost_params.mean_cost
        results['k_ne'][i] = a['k_ne']
        results['k_opt'][i] = a['k_opt']
        results['k_stackelberg'][i] = a['k_stackelberg']
        results['p_star'][i] = a['p_stackelberg']
        results['total_incentive'][i] = a['incentive_cost']
        results['welfare_ne'][i] = a['welfare_ne']
        results['welfare_opt'][i] = a['welfare_opt']
        results['welfare_stackelberg'][i] = a['welfare_stackelberg']
        results['poa'][i] = a['welfare_opt'] / a['welfare_ne'] if a['welfare_ne'] > 0 else np.inf
        results['efficiency_gain'][i] = a['efficiency_gain']
    
    return results


def compare_incentives_homogeneous_heterogeneous(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float,
    spread_ratio: float = 2.0
) -> Dict:
    """
    Compare incentive mechanisms: homogeneous vs heterogeneous with same mean.
    
    Parameters
    ----------
    N, R, L, B : parameters
    c : float
        Homogeneous cost (and mean for heterogeneous)
    spread_ratio : float
        c_max/c_min for heterogeneous case
    
    Returns
    -------
    comparison : dict
    """
    rho = compute_rho(R, L)
    
    # Homogeneous
    hom = analyze_stackelberg(N, R, L, B, c, use_robust=True)
    
    # Heterogeneous with same mean
    c_min = 2 * c / (1 + spread_ratio)
    c_max = spread_ratio * c_min
    cost_params = HeterogeneousCostParams(c_min=c_min, c_max=c_max)
    het = analyze_stackelberg_heterogeneous(N, rho, B, cost_params)
    
    return {
        'homogeneous': hom,
        'heterogeneous': het,
        'p_star_diff': het['p_stackelberg'] - hom['p_stackelberg'],
        'budget_diff': het['incentive_cost'] - hom['incentive_cost'],
        'k_diff': het['k_stackelberg'] - hom['k_stackelberg'],
        'efficiency_gain_diff': het['efficiency_gain'] - hom['efficiency_gain'],
    }