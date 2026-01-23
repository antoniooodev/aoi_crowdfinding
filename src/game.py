"""Game-theoretic model utilities.

Implements utilities, equilibrium/optimum computation, welfare metrics, and Price of Anarchy calculations for volunteer participation in the crowd-finding model.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, Optional
from .spatial import analytical_coverage_prob, compute_rho
from .aoi import expected_aoi_from_k


def benefit_function(P_det: float, B: float) -> float:
    """
    Compute benefit from detection probability.
    
    f(Δ) = B * P_det  (simplified form)
    
    Parameters
    ----------
    P_det : float
        Detection probability
    B : float
        Maximum benefit
    
    Returns
    -------
    benefit : float
        Benefit value
    """
    return B * P_det


def utility_active(k: int, R: float, L: float, B: float, c: float) -> float:
    """
    Compute utility for an active volunteer when k total are active.
    
    U(active; k) = B * P_det(k) - c
    
    Parameters
    ----------
    k : int
        Total number of active volunteers (including self)
    R : float
        Detection radius
    L : float
        Area side length
    B : float
        Benefit parameter
    c : float
        Participation cost
    
    Returns
    -------
    utility : float
        Utility for active volunteer
    """
    P_det = analytical_coverage_prob(k, R, L)
    return B * P_det - c


def utility_inactive(k: int, R: float, L: float, B: float) -> float:
    """
    Compute utility for an inactive volunteer when k are active.
    
    U(inactive; k) = B * P_det(k)
    
    Parameters
    ----------
    k : int
        Number of active volunteers (not including self)
    R : float
        Detection radius
    L : float
        Area side length
    B : float
        Benefit parameter
    
    Returns
    -------
    utility : float
        Utility for inactive volunteer
    """
    P_det = analytical_coverage_prob(k, R, L)
    return B * P_det


def marginal_utility(k: int, R: float, L: float, B: float, c: float) -> float:
    """
    Compute marginal utility of participation.
    
    ΔU(k) = U(active; k+1) - U(inactive; k) = B * [P_det(k+1) - P_det(k)] - c
    
    Parameters
    ----------
    k : int
        Number of other active volunteers
    R : float
        Detection radius
    L : float
        Area side length
    B : float
        Benefit parameter
    c : float
        Participation cost
    
    Returns
    -------
    marginal : float
        Marginal utility of becoming active
    """
    rho = compute_rho(R, L)
    marginal_detection = rho * (1 - rho) ** k
    return B * marginal_detection - c


def find_nash_equilibrium(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> int:
    """
    Find symmetric Nash equilibrium number of active volunteers.
    
    Parameters
    ----------
    N : int
        Total number of volunteers
    R : float
        Detection radius
    L : float
        Area side length
    B : float
        Benefit parameter
    c : float
        Participation cost
    
    Returns
    -------
    k_star : int
        NE number of active volunteers
    """
    rho = compute_rho(R, L)
    
    # Check boundary cases
    if c <= 0:
        return N
    if c >= B * rho:
        return 0
    
    # Find k* such that B*rho*(1-rho)^(k*-1) >= c > B*rho*(1-rho)^k*
    # Solving: k* = floor(1 + ln(c/(B*rho)) / ln(1-rho))
    
    k_star_continuous = 1 + np.log(c / (B * rho)) / np.log(1 - rho)
    k_star = int(np.floor(k_star_continuous))
    
    # Clamp to valid range
    k_star = max(0, min(N, k_star))
    
    return k_star


def find_nash_equilibrium_search(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> int:
    """
    Find NE by exhaustive search (verification method).
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    
    Returns
    -------
    k_star : int
        NE number of active volunteers
    """
    for k in range(N + 1):
        # Check if k is a NE
        # Condition 1: Active volunteers don't want to deviate
        if k > 0:
            u_active = utility_active(k, R, L, B, c)
            u_deviate_inactive = utility_inactive(k - 1, R, L, B)
            if u_active < u_deviate_inactive - 1e-12:
                continue
        
        # Condition 2: Inactive volunteers don't want to deviate
        if k < N:
            u_inactive = utility_inactive(k, R, L, B)
            u_deviate_active = utility_active(k + 1, R, L, B, c)
            if u_inactive < u_deviate_active - 1e-12:
                continue
        
        return k
    
    return 0  # Fallback


def social_welfare(k: int, N: int, R: float, L: float, B: float, c: float) -> float:
    """
    Compute social welfare with k active volunteers.
    
    W(k) = N * B * P_det(k) - k * c
    
    Parameters
    ----------
    k : int
        Number of active volunteers
    N : int
        Total volunteers
    R, L, B, c : parameters
    
    Returns
    -------
    welfare : float
        Total social welfare
    """
    P_det = analytical_coverage_prob(k, R, L)
    return N * B * P_det - k * c


def find_social_optimum(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> int:
    """
    Find socially optimal number of active volunteers.
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    
    Returns
    -------
    k_opt : int
        Socially optimal active volunteers
    """
    rho = compute_rho(R, L)
    
    # Check boundary
    if c >= N * B * rho:
        return 0
    if c <= 0:
        return N
    
    # Analytical solution: k_opt = floor(1 + ln(c/(N*B*rho)) / ln(1-rho))
    k_opt_continuous = 1 + np.log(c / (N * B * rho)) / np.log(1 - rho)
    k_opt = int(np.floor(k_opt_continuous))
    
    # Clamp to valid range
    k_opt = max(0, min(N, k_opt))
    
    return k_opt


def find_social_optimum_search(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> int:
    """
    Find social optimum by exhaustive search.
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    
    Returns
    -------
    k_opt : int
        Socially optimal active volunteers
    """
    welfare_values = [social_welfare(k, N, R, L, B, c) for k in range(N + 1)]
    return int(np.argmax(welfare_values))


def price_of_anarchy(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> float:
    """
    Compute Price of Anarchy.
    
    PoA = W(k_opt) / W(k*)
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    
    Returns
    -------
    poa : float
        Price of Anarchy (>= 1, or inf if W(k*)=0)
    """
    k_star = find_nash_equilibrium(N, R, L, B, c)
    k_opt = find_social_optimum(N, R, L, B, c)
    
    W_star = social_welfare(k_star, N, R, L, B, c)
    W_opt = social_welfare(k_opt, N, R, L, B, c)
    
    if W_star <= 0:
        return np.inf
    
    return W_opt / W_star


def participation_gap(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> int:
    """
    Compute participation gap between social optimum and NE.
    
    Δk = k_opt - k*
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    
    Returns
    -------
    gap : int
        Participation gap
    """
    k_star = find_nash_equilibrium(N, R, L, B, c)
    k_opt = find_social_optimum(N, R, L, B, c)
    return k_opt - k_star


def analyze_equilibrium(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> Dict:
    """
    Complete equilibrium analysis.
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    
    Returns
    -------
    results : dict
        Dictionary containing all equilibrium metrics
    """
    rho = compute_rho(R, L)
    
    k_star = find_nash_equilibrium(N, R, L, B, c)
    k_opt = find_social_optimum(N, R, L, B, c)
    
    P_det_star = analytical_coverage_prob(k_star, R, L)
    P_det_opt = analytical_coverage_prob(k_opt, R, L)
    
    aoi_star = expected_aoi_from_k(k_star, R, L)
    aoi_opt = expected_aoi_from_k(k_opt, R, L)
    
    W_star = social_welfare(k_star, N, R, L, B, c)
    W_opt = social_welfare(k_opt, N, R, L, B, c)
    
    poa = W_opt / W_star if W_star > 0 else np.inf
    
    return {
        'N': N,
        'R': R,
        'L': L,
        'B': B,
        'c': c,
        'rho': rho,
        'B_rho': B * rho,
        'NB_rho': N * B * rho,
        'k_star': k_star,
        'k_opt': k_opt,
        'participation_gap': k_opt - k_star,
        'P_det_star': P_det_star,
        'P_det_opt': P_det_opt,
        'aoi_star': aoi_star,
        'aoi_opt': aoi_opt,
        'welfare_star': W_star,
        'welfare_opt': W_opt,
        'price_of_anarchy': poa,
    }


def sweep_cost(
    N: int,
    R: float,
    L: float,
    B: float,
    c_values: NDArray[np.float64]
) -> Dict[str, NDArray]:
    """
    Analyze equilibrium across range of cost values.
    
    Parameters
    ----------
    N, R, L, B : parameters
    c_values : array
        Cost values to analyze
    
    Returns
    -------
    results : dict of arrays
    """
    n = len(c_values)
    results = {
        'c': c_values.copy(),
        'k_ne': np.zeros(n, dtype=int),
        'k_opt': np.zeros(n, dtype=int),
        'welfare_ne': np.zeros(n),
        'welfare_opt': np.zeros(n),
        'poa': np.zeros(n),
        'gap': np.zeros(n, dtype=int),
    }
    
    for i, c in enumerate(c_values):
        analysis = analyze_equilibrium(N, R, L, B, c)
        results['k_ne'][i] = analysis['k_star']
        results['k_opt'][i] = analysis['k_opt']
        results['welfare_ne'][i] = analysis['welfare_star']
        results['welfare_opt'][i] = analysis['welfare_opt']
        results['poa'][i] = analysis['price_of_anarchy']
        results['gap'][i] = analysis['participation_gap']
    
    return results


# =============================================================================
# HETEROGENEOUS COST MODEL EXTENSIONS (Version 2.0)
# =============================================================================

from .config import HeterogeneousCostParams


def cost_threshold(k: int, rho: float, B: float) -> float:
    """
    Marginal benefit function φ(k) = Bρ(1-ρ)^(k-1).
    
    This is the benefit from the k-th volunteer joining when k-1 others are active.
    
    Parameters
    ----------
    k : int
        Number of active volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    
    Returns
    -------
    threshold : float
        Marginal benefit value
    """
    if k <= 0:
        return B * rho  # Maximum benefit when no one is active
    return B * rho * (1 - rho) ** (k - 1)


def nash_threshold(k: int, rho: float, B: float) -> float:
    """
    Nash equilibrium cost threshold c̄(k) = Bρ(1-ρ)^k.
    
    When k volunteers are active, the threshold for participation is c̄(k).
    A volunteer with cost c_i participates iff c_i <= c̄(k).
    
    Parameters
    ----------
    k : int
        Number of active volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    
    Returns
    -------
    threshold : float
        Cost threshold
    """
    if k < 0:
        return B * rho
    return B * rho * (1 - rho) ** k


def find_nash_heterogeneous(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams,
    tol: float = 1e-8,
    max_iter: int = 100
) -> Tuple[int, float]:
    """
    Find Nash equilibrium with heterogeneous costs.
    
    Fixed-point equation: k* = N · F(Bρ(1-ρ)^k*)
    
    Handles three cases:
    1. Corner k*=N: if c_max <= Bρ(1-ρ)^(N-1), all participate
    2. Corner k*=0: if c_min >= Bρ, no one participates  
    3. Interior: fixed-point iteration with damping
    
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
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    k_star : int
        Equilibrium number of active volunteers
    c_bar_star : float
        Equilibrium cost threshold
    """
    c_min = cost_params.c_min
    c_max = cost_params.c_max
    
    # Handle homogeneous case
    if cost_params.is_homogeneous:
        c = c_min
        if c >= B * rho:
            return 0, 0.0
        if c <= 0:
            return N, B * rho * (1 - rho) ** (N - 1)
        k_star = int(np.floor(1 + np.log(c / (B * rho)) / np.log(1 - rho)))
        k_star = max(0, min(N, k_star))
        return k_star, nash_threshold(k_star, rho, B)
    
    # Corner case 1: No one participates
    # If c_min >= Bρ, even the cheapest volunteer won't join
    c_bar_0 = B * rho  # Threshold when k=0
    if c_min >= c_bar_0:
        return 0, c_bar_0
    
    # Corner case 2: Everyone participates
    # If c_max <= Bρ(1-ρ)^(N-1), even the most expensive will join
    c_bar_N = B * rho * (1 - rho) ** (N - 1)  # Threshold for N-th volunteer
    if c_max <= c_bar_N:
        return N, nash_threshold(N, rho, B)
    
    # Interior equilibrium: use damped fixed-point iteration
    # k* solves: k = N · F(Bρ(1-ρ)^k)
    
    # Better initial guess: find k where threshold equals mean cost
    mean_cost = cost_params.mean_cost
    if mean_cost < B * rho and mean_cost > 0:
        # Solve Bρ(1-ρ)^k = mean_cost for initial k
        k = np.log(mean_cost / (B * rho)) / np.log(1 - rho)
        k = max(0, min(N, k))
    else:
        k = N / 2
    
    # Damped iteration to prevent oscillation
    damping = 0.5
    k_prev = k
    
    for iteration in range(max_iter):
        # Compute threshold for current k
        k_int = int(round(k))
        k_int = max(0, min(N, k_int))
        c_bar = B * rho * (1 - rho) ** k_int
        
        # Compute CDF value (fraction with cost <= threshold)
        F_c_bar = cost_params.cdf(c_bar)
        
        # New k from fixed-point equation
        k_new = N * F_c_bar
        
        # Apply damping: k_next = (1-α)·k + α·k_new
        k_next = (1 - damping) * k + damping * k_new
        
        # Check convergence
        if abs(k_next - k) < tol and abs(k_next - k_prev) < tol:
            break
        
        k_prev = k
        k = k_next
    
    # Round to integer and verify
    k_star = int(round(k))
    k_star = max(0, min(N, k_star))
    
    # Final verification: check if this is actually an equilibrium
    # At equilibrium: agents with c_i <= c̄(k*) participate
    c_bar_star = nash_threshold(k_star, rho, B)
    k_check = int(round(N * cost_params.cdf(c_bar_star)))
    
    # If verification fails, search for stable equilibrium
    if abs(k_check - k_star) > 1:
        # Brute force search for stable point
        best_k = 0
        best_diff = float('inf')
        
        for k_test in range(N + 1):
            c_bar_test = nash_threshold(k_test, rho, B)
            k_implied = N * cost_params.cdf(c_bar_test)
            diff = abs(k_implied - k_test)
            
            if diff < best_diff:
                best_diff = diff
                best_k = k_test
        
        k_star = best_k
        c_bar_star = nash_threshold(k_star, rho, B)
    
    return k_star, c_bar_star

def find_nash_heterogeneous_with_costs(
    N: int,
    rho: float,
    B: float,
    costs: NDArray[np.float64]
) -> Tuple[int, float, NDArray[np.bool_]]:
    """
    Find Nash equilibrium given realized costs.
    
    Parameters
    ----------
    N : int
        Number of volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    costs : ndarray of shape (N,)
        Realized costs for each volunteer
    
    Returns
    -------
    k_star : int
        Equilibrium participation count
    c_bar_star : float
        Equilibrium threshold
    active : ndarray of bool
        Which volunteers participate
    """
    # Sort costs to find threshold
    sorted_costs = np.sort(costs)
    
    # Find k such that:
    # - The k-th lowest cost volunteer is willing: c_{(k)} <= Bρ(1-ρ)^(k-1)
    # - The (k+1)-th is not: c_{(k+1)} > Bρ(1-ρ)^k
    
    k_star = 0
    for k in range(1, N + 1):
        threshold = nash_threshold(k - 1, rho, B)  # Threshold when k-1 others active
        if sorted_costs[k - 1] <= threshold:
            k_star = k
        else:
            break
    
    c_bar_star = nash_threshold(k_star, rho, B)
    active = costs <= c_bar_star
    
    return k_star, c_bar_star, active


def verify_nash_heterogeneous(
    k: int,
    rho: float,
    B: float,
    costs: NDArray[np.float64],
    active: NDArray[np.bool_]
) -> bool:
    """
    Verify that given participation is a Nash equilibrium.
    
    Parameters
    ----------
    k : int
        Number of active volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    costs : ndarray
        Volunteer costs
    active : ndarray of bool
        Which volunteers are active
    
    Returns
    -------
    is_ne : bool
        True if this is a Nash equilibrium
    """
    N = len(costs)
    
    for i in range(N):
        c_i = costs[i]
        
        if active[i]:
            # Active volunteer: check they don't want to defect
            # Utility active = B*P_det(k) - c_i
            # Utility inactive = B*P_det(k-1)
            # Want: c_i <= B*(P_det(k) - P_det(k-1)) = B*ρ*(1-ρ)^(k-1)
            marginal_benefit = B * rho * (1 - rho) ** (k - 1)
            if c_i > marginal_benefit + 1e-10:
                return False
        else:
            # Inactive volunteer: check they don't want to join
            # Utility inactive = B*P_det(k)
            # Utility active = B*P_det(k+1) - c_i
            # Want: c_i >= B*(P_det(k+1) - P_det(k)) = B*ρ*(1-ρ)^k
            marginal_benefit = B * rho * (1 - rho) ** k
            if c_i < marginal_benefit - 1e-10:
                return False
    
    return True


def social_welfare_heterogeneous(
    k: int,
    N: int,
    rho: float,
    B: float,
    cost_sum: float
) -> float:
    """
    Compute social welfare with k active volunteers (heterogeneous costs).
    
    W(k) = N · B · P_det(k) - Σ_{i active} c_i
    
    Parameters
    ----------
    k : int
        Number of active volunteers
    N : int
        Total volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    cost_sum : float
        Sum of costs of active volunteers
    
    Returns
    -------
    welfare : float
    """
    P_det = 1 - (1 - rho) ** k if k > 0 else 0.0
    return N * B * P_det - cost_sum


def find_social_optimum_heterogeneous(
    N: int,
    rho: float,
    B: float,
    costs: NDArray[np.float64]
) -> Tuple[int, NDArray[np.bool_], float]:
    """
    Find socially optimal selection given realized costs.
    
    Optimal rule: Select the k volunteers with lowest costs such that
    the marginal benefit exceeds the marginal cost.
    
    Parameters
    ----------
    N : int
        Number of volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    costs : ndarray of shape (N,)
        Realized costs
    
    Returns
    -------
    k_opt : int
        Optimal participation count
    active : ndarray of bool
        Which volunteers should participate
    welfare : float
        Optimal welfare
    """
    # Sort costs
    sorted_indices = np.argsort(costs)
    sorted_costs = costs[sorted_indices]
    
    # Greedy selection: add volunteers in order of cost until marginal benefit < cost
    k_opt = 0
    total_cost = 0.0
    
    for k in range(1, N + 1):
        # Marginal benefit of k-th volunteer
        marginal_benefit = N * B * rho * (1 - rho) ** (k - 1)
        marginal_cost = sorted_costs[k - 1]
        
        if marginal_benefit >= marginal_cost:
            k_opt = k
            total_cost += marginal_cost
        else:
            break
    
    # Determine which volunteers are active
    active = np.zeros(N, dtype=bool)
    active[sorted_indices[:k_opt]] = True
    
    # Compute welfare
    welfare = social_welfare_heterogeneous(k_opt, N, rho, B, total_cost)
    
    return k_opt, active, welfare


def find_social_optimum_heterogeneous_expected(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams
) -> Tuple[int, float]:
    """
    Find expected socially optimal participation level.
    
    Uses order statistics: E[c_{(k)}] ≈ F^{-1}(k/N) for large N.
    
    Parameters
    ----------
    N, rho, B : parameters
    cost_params : HeterogeneousCostParams
    
    Returns
    -------
    k_opt : int
        Expected optimal participation
    c_bar_opt : float
        Optimal cost threshold
    """
    # Find k such that N·B·ρ·(1-ρ)^(k-1) >= E[c_{(k)}]
    k_opt = 0
    
    for k in range(1, N + 1):
        marginal_benefit = N * B * rho * (1 - rho) ** (k - 1)
        expected_kth_cost = cost_params.expected_kth_order_statistic(k, N)
        
        if marginal_benefit >= expected_kth_cost:
            k_opt = k
        else:
            break
    
    c_bar_opt = cost_params.quantile(k_opt / N) if k_opt > 0 else cost_params.c_min
    
    return k_opt, c_bar_opt


def expected_welfare_heterogeneous(
    k: int,
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams
) -> float:
    """
    Expected welfare when k lowest-cost volunteers participate.
    
    W(k) = N·B·P_det(k) - E[Σ_{j=1}^k c_{(j)}]
    
    Parameters
    ----------
    k : int
        Number of participants
    N : int
        Total volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    cost_params : HeterogeneousCostParams
        Cost distribution
    
    Returns
    -------
    welfare : float
        Expected welfare
    """
    if k <= 0:
        return 0.0
    
    P_det = 1 - (1 - rho) ** k
    expected_cost_sum = cost_params.expected_sum_of_k_lowest(k, N)
    
    return N * B * P_det - expected_cost_sum


def price_of_anarchy_heterogeneous(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams
) -> float:
    """
    Compute expected Price of Anarchy with heterogeneous costs.
    
    PoA = E[W(k_opt)] / E[W(k*)]
    
    Parameters
    ----------
    N, rho, B : parameters
    cost_params : HeterogeneousCostParams
    
    Returns
    -------
    poa : float
        Price of Anarchy
    """
    k_star, _ = find_nash_heterogeneous(N, rho, B, cost_params)
    k_opt, _ = find_social_optimum_heterogeneous_expected(N, rho, B, cost_params)
    
    W_star = expected_welfare_heterogeneous(k_star, N, rho, B, cost_params)
    W_opt = expected_welfare_heterogeneous(k_opt, N, rho, B, cost_params)
    
    if W_star <= 0:
        return np.inf
    
    return W_opt / W_star


def participation_gap_heterogeneous(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams
) -> int:
    """
    Compute participation gap with heterogeneous costs.
    
    Parameters
    ----------
    N, rho, B : parameters
    cost_params : HeterogeneousCostParams
    
    Returns
    -------
    gap : int
        k_opt - k*
    """
    k_star, _ = find_nash_heterogeneous(N, rho, B, cost_params)
    k_opt, _ = find_social_optimum_heterogeneous_expected(N, rho, B, cost_params)
    return k_opt - k_star


def analyze_equilibrium_heterogeneous(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams
) -> Dict:
    """
    Complete equilibrium analysis with heterogeneous costs.
    
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
    results : dict
        Comprehensive analysis results
    """
    k_star, c_bar_star = find_nash_heterogeneous(N, rho, B, cost_params)
    k_opt, c_bar_opt = find_social_optimum_heterogeneous_expected(N, rho, B, cost_params)
    
    P_det_star = 1 - (1 - rho) ** k_star if k_star > 0 else 0.0
    P_det_opt = 1 - (1 - rho) ** k_opt if k_opt > 0 else 0.0
    
    W_star = expected_welfare_heterogeneous(k_star, N, rho, B, cost_params)
    W_opt = expected_welfare_heterogeneous(k_opt, N, rho, B, cost_params)
    
    poa = W_opt / W_star if W_star > 0 else np.inf
    
    return {
        'N': N,
        'rho': rho,
        'B': B,
        'B_rho': B * rho,
        'NB_rho': N * B * rho,
        'c_min': cost_params.c_min,
        'c_max': cost_params.c_max,
        'mean_cost': cost_params.mean_cost,
        'heterogeneity_ratio': cost_params.heterogeneity_ratio,
        'distribution': cost_params.distribution,
        # Nash equilibrium
        'k_star': k_star,
        'c_bar_star': c_bar_star,
        'P_det_star': P_det_star,
        'welfare_star': W_star,
        # Social optimum
        'k_opt': k_opt,
        'c_bar_opt': c_bar_opt,
        'P_det_opt': P_det_opt,
        'welfare_opt': W_opt,
        # Comparison
        'participation_gap': k_opt - k_star,
        'price_of_anarchy': poa,
    }


def sweep_heterogeneity(
    N: int,
    rho: float,
    B: float,
    mean_cost: float,
    spread_ratios: NDArray[np.float64]
) -> Dict[str, NDArray]:
    """
    Analyze equilibrium across heterogeneity levels.
    
    Parameters
    ----------
    N, rho, B : parameters
    mean_cost : float
        Fixed mean cost
    spread_ratios : array
        c_max/c_min ratios to test
    
    Returns
    -------
    results : dict of arrays
    """
    n = len(spread_ratios)
    results = {
        'spread_ratio': spread_ratios.copy(),
        'c_min': np.zeros(n),
        'c_max': np.zeros(n),
        'k_star': np.zeros(n, dtype=int),
        'k_opt': np.zeros(n, dtype=int),
        'gap': np.zeros(n, dtype=int),
        'poa': np.zeros(n),
        'welfare_star': np.zeros(n),
        'welfare_opt': np.zeros(n),
    }
    
    for i, ratio in enumerate(spread_ratios):
        # Compute c_min, c_max from mean and ratio
        c_min = 2 * mean_cost / (1 + ratio)
        c_max = ratio * c_min
        
        cost_params = HeterogeneousCostParams(c_min=c_min, c_max=c_max)
        analysis = analyze_equilibrium_heterogeneous(N, rho, B, cost_params)
        
        results['c_min'][i] = c_min
        results['c_max'][i] = c_max
        results['k_star'][i] = analysis['k_star']
        results['k_opt'][i] = analysis['k_opt']
        results['gap'][i] = analysis['participation_gap']
        results['poa'][i] = analysis['price_of_anarchy']
        results['welfare_star'][i] = analysis['welfare_star']
        results['welfare_opt'][i] = analysis['welfare_opt']
    
    return results


def sweep_cost_heterogeneous(
    N: int,
    rho: float,
    B: float,
    c_min_values: NDArray[np.float64],
    c_max_values: NDArray[np.float64]
) -> Dict[str, NDArray]:
    """
    Analyze equilibrium across cost ranges.
    
    Parameters
    ----------
    N, rho, B : parameters
    c_min_values, c_max_values : arrays
        Corresponding min/max costs
    
    Returns
    -------
    results : dict of arrays
    """
    n = len(c_min_values)
    results = {
        'c_min': c_min_values.copy(),
        'c_max': c_max_values.copy(),
        'mean_cost': np.zeros(n),
        'k_star': np.zeros(n, dtype=int),
        'k_opt': np.zeros(n, dtype=int),
        'gap': np.zeros(n, dtype=int),
        'poa': np.zeros(n),
    }
    
    for i in range(n):
        cost_params = HeterogeneousCostParams(
            c_min=c_min_values[i], 
            c_max=c_max_values[i]
        )
        analysis = analyze_equilibrium_heterogeneous(N, rho, B, cost_params)
        
        results['mean_cost'][i] = cost_params.mean_cost
        results['k_star'][i] = analysis['k_star']
        results['k_opt'][i] = analysis['k_opt']
        results['gap'][i] = analysis['participation_gap']
        results['poa'][i] = analysis['price_of_anarchy']
    
    return results


def compare_homogeneous_heterogeneous(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float,
    spread_ratio: float = 2.0
) -> Dict:
    """
    Compare homogeneous vs heterogeneous model with same mean cost.
    
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
    
    # Homogeneous analysis
    hom = analyze_equilibrium(N, R, L, B, c)
    
    # Heterogeneous analysis with same mean
    c_min = 2 * c / (1 + spread_ratio)
    c_max = spread_ratio * c_min
    cost_params = HeterogeneousCostParams(c_min=c_min, c_max=c_max)
    het = analyze_equilibrium_heterogeneous(N, rho, B, cost_params)
    
    return {
        'homogeneous': hom,
        'heterogeneous': het,
        'k_star_diff': het['k_star'] - hom['k_star'],
        'k_opt_diff': het['k_opt'] - hom['k_opt'],
        'gap_diff': het['participation_gap'] - hom['participation_gap'],
        'poa_diff': het['price_of_anarchy'] - hom['price_of_anarchy'],
    }