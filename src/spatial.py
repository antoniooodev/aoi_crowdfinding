"""
spatial.py - 2D spatial computations for crowd-finding

CORRECTION APPLIED: Target sampling in interior [R, L-R]² to match
analytical assumption that ρ = πR²/L² is constant (no boundary truncation).
"""
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional


def generate_positions(
    n: int,
    L: float,
    seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Generate n uniformly distributed positions in [0, L]².
    
    Parameters
    ----------
    n : int
        Number of positions to generate
    L : float
        Side length of square area
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    positions : ndarray of shape (n, 2)
        Array of (x, y) coordinates
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, L, size=(n, 2))


def generate_target(
    L: float,
    R: float = 0.0,
    interior: bool = True,
    seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Generate single target position.
    
    CORRECTION: When interior=True, samples from [R, L-R]² to eliminate
    boundary truncation effects, ensuring ρ = πR²/L² is exact.
    
    Parameters
    ----------
    L : float
        Side length of square area
    R : float
        Detection radius (used for interior sampling)
    interior : bool
        If True, sample from [R, L-R]² to avoid boundary effects
    seed : int, optional
        Random seed
    
    Returns
    -------
    target : ndarray of shape (2,)
        Target (x, y) coordinates
    """
    rng = np.random.default_rng(seed)
    
    if interior and R > 0 and 2 * R < L:
        # Sample from interior to avoid boundary truncation
        return rng.uniform(R, L - R, size=2)
    else:
        # Full domain sampling (original behavior)
        return rng.uniform(0, L, size=2)


def compute_distances(
    positions: NDArray[np.float64],
    target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute Euclidean distances from each position to target.
    
    Parameters
    ----------
    positions : ndarray of shape (n, 2)
        Volunteer positions
    target : ndarray of shape (2,)
        Target position
    
    Returns
    -------
    distances : ndarray of shape (n,)
        Distance from each volunteer to target
    """
    return np.linalg.norm(positions - target, axis=1)


def coverage_mask(
    distances: NDArray[np.float64],
    R: float
) -> NDArray[np.bool_]:
    """
    Determine which volunteers cover the target.
    
    Parameters
    ----------
    distances : ndarray of shape (n,)
        Distances to target
    R : float
        Detection radius
    
    Returns
    -------
    mask : ndarray of shape (n,) of bool
        True if volunteer covers target
    """
    return distances <= R


def count_covering(
    positions: NDArray[np.float64],
    target: NDArray[np.float64],
    R: float
) -> int:
    """
    Count number of volunteers covering the target.
    
    Parameters
    ----------
    positions : ndarray of shape (n, 2)
        Volunteer positions
    target : ndarray of shape (2,)
        Target position
    R : float
        Detection radius
    
    Returns
    -------
    count : int
        Number of covering volunteers
    """
    distances = compute_distances(positions, target)
    return int(np.sum(coverage_mask(distances, R)))


def analytical_coverage_prob(k: int, R: float, L: float) -> float:
    """
    Compute analytical coverage probability with k active volunteers.
    
    P_det(k) = 1 - (1 - ρ)^k  where ρ = πR²/L²
    
    NOTE: This formula assumes no boundary effects (target in interior).
    
    Parameters
    ----------
    k : int
        Number of active volunteers
    R : float
        Detection radius
    L : float
        Area side length
    
    Returns
    -------
    prob : float
        Probability of at least one volunteer covering target
    """
    if k <= 0:
        return 0.0
    rho = np.pi * R**2 / L**2
    return 1.0 - (1.0 - rho) ** k


def empirical_coverage_prob(
    k: int,
    R: float,
    L: float,
    n_samples: int = 10000,
    seed: Optional[int] = None,
    interior: bool = True
) -> Tuple[float, float]:
    """
    Estimate coverage probability via Monte Carlo simulation.
    
    CORRECTION: Uses interior target sampling by default to match
    analytical formula assumptions.
    
    Parameters
    ----------
    k : int
        Number of active volunteers
    R : float
        Detection radius
    L : float
        Area side length
    n_samples : int
        Number of Monte Carlo samples
    seed : int, optional
        Random seed
    interior : bool
        If True, sample target from interior [R, L-R]²
    
    Returns
    -------
    mean : float
        Estimated coverage probability
    std_err : float
        Standard error of estimate
    """
    if k <= 0:
        return 0.0, 0.0
    
    rng = np.random.default_rng(seed)
    successes = 0
    
    for _ in range(n_samples):
        # Volunteers always uniform in [0, L]²
        positions = rng.uniform(0, L, size=(k, 2))
        
        # Target in interior or full domain
        if interior and R > 0 and 2 * R < L:
            target = rng.uniform(R, L - R, size=2)
        else:
            target = rng.uniform(0, L, size=2)
        
        distances = np.linalg.norm(positions - target, axis=1)
        if np.any(distances <= R):
            successes += 1
    
    mean = successes / n_samples
    std_err = np.sqrt(mean * (1 - mean) / n_samples)
    return mean, std_err


def coverage_probability_vectorized(
    k_values: NDArray[np.int_],
    R: float,
    L: float
) -> NDArray[np.float64]:
    """
    Vectorized analytical coverage probability for array of k values.
    
    Parameters
    ----------
    k_values : ndarray of int
        Array of active volunteer counts
    R : float
        Detection radius
    L : float
        Area side length
    
    Returns
    -------
    probs : ndarray of float
        Coverage probabilities for each k
    """
    rho = np.pi * R**2 / L**2
    return np.where(k_values > 0, 1.0 - (1.0 - rho) ** k_values, 0.0)


def compute_rho(R: float, L: float) -> float:
    """
    Compute coverage ratio ρ = πR²/L².
    
    Parameters
    ----------
    R : float
        Detection radius
    L : float
        Area side length
    
    Returns
    -------
    rho : float
        Coverage ratio
    """
    return np.pi * R**2 / L**2


def compute_thresholds(N: int, R: float, L: float, B: float) -> dict:
    """
    Compute key thresholds for the game.
    
    These thresholds determine the cost ranges where k* > 0 and k_opt > 0.
    
    Parameters
    ----------
    N : int
        Number of volunteers
    R : float
        Detection radius
    L : float
        Area side length
    B : float
        Benefit parameter
    
    Returns
    -------
    thresholds : dict
        Dictionary with threshold values
    """
    rho = compute_rho(R, L)
    B_rho = B * rho
    NB_rho = N * B * rho
    
    return {
        'rho': rho,
        'B_rho': B_rho,           # Threshold for k* > 0: need c < B*rho
        'NB_rho': NB_rho,         # Threshold for k_opt > 0: need c < N*B*rho
        'c_max_ne': B_rho,        # Max cost for non-zero NE
        'c_max_opt': NB_rho,      # Max cost for non-zero social optimum
    }
