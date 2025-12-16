"""
game.py - Game theory computations for crowd-finding
"""
import numpy as np
from numpy.typing import NDArray
from typing import Dict
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
