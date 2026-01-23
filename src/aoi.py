"""Age of Information (AoI) utilities.

Provides closed-form AoI expressions from detection probability, vectorized helpers, and a simple trajectory simulator for validating analytical results.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .spatial import analytical_coverage_prob, compute_rho


def expected_aoi(P_det: float) -> float:
    """
    Compute expected time-average AoI given detection probability.
    
    E[Δ] = 1/P_det - 1
    
    Parameters
    ----------
    P_det : float
        Detection probability per slot
    
    Returns
    -------
    aoi : float
        Expected time-average AoI
    """
    if P_det <= 0:
        return np.inf
    if P_det >= 1:
        return 0.0
    return 1.0 / P_det - 1.0


def expected_aoi_from_k(k: int, R: float, L: float) -> float:
    """
    Compute expected AoI for k active volunteers.
    
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
    aoi : float
        Expected time-average AoI
    """
    P_det = analytical_coverage_prob(k, R, L)
    return expected_aoi(P_det)


def aoi_vectorized(k_values: NDArray[np.int_], R: float, L: float) -> NDArray[np.float64]:
    """
    Vectorized AoI computation for array of k values.
    
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
    aoi_values : ndarray of float
        Expected AoI for each k
    """
    rho = compute_rho(R, L)
    P_det = np.where(k_values > 0, 1.0 - (1.0 - rho) ** k_values, 0.0)
    return np.where(P_det > 0, 1.0 / P_det - 1.0, np.inf)


def simulate_aoi_trajectory(
    P_det: float,
    T: int,
    seed: Optional[int] = None
) -> NDArray[np.int_]:
    """
    Simulate AoI trajectory over T time slots.
    
    Parameters
    ----------
    P_det : float
        Detection probability per slot
    T : int
        Number of time slots
    seed : int, optional
        Random seed
    
    Returns
    -------
    trajectory : ndarray of shape (T,)
        AoI value at each time slot
    """
    rng = np.random.default_rng(seed)
    trajectory = np.zeros(T, dtype=np.int64)
    
    current_aoi = 0
    for t in range(T):
        if rng.random() < P_det:
            current_aoi = 0
        else:
            current_aoi += 1
        trajectory[t] = current_aoi
    
    return trajectory

def simulate_aoi_trajectory_fast(
    P_det: float,
    T: int,
    seed: Optional[int] = None
) -> NDArray[np.int_]:
    """Simulate AoI trajectory over T time slots (fast path).

    This function is API-compatible with `simulate_aoi_trajectory` and
    uses the same random number generator semantics, so that using the
    same seed produces the same trajectory.

    Parameters
    ----------
    P_det : float
        Detection probability per slot
    T : int
        Number of time slots
    seed : int, optional
        Random seed

    Returns
    -------
    trajectory : ndarray of shape (T,)
        AoI value at each time slot
    """
    
    return simulate_aoi_trajectory(P_det, T, seed=seed)



def time_average_aoi(trajectory: NDArray[np.int_]) -> float:
    """
    Compute time-average AoI from trajectory.
    
    Parameters
    ----------
    trajectory : ndarray
        AoI values over time
    
    Returns
    -------
    avg_aoi : float
        Time-average AoI
    """
    return float(np.mean(trajectory))


def peak_aoi(trajectory: NDArray[np.int_]) -> int:
    """
    Compute peak (maximum) AoI from trajectory.
    
    Parameters
    ----------
    trajectory : ndarray
        AoI values over time
    
    Returns
    -------
    peak : int
        Maximum AoI value
    """
    return int(np.max(trajectory))


def marginal_aoi_reduction(k: int, R: float, L: float) -> float:
    """
    Compute marginal AoI reduction from adding k-th volunteer.
    
    δΔ(k) = Δ(k-1) - Δ(k)
    
    Parameters
    ----------
    k : int
        Number of active volunteers (including the new one)
    R : float
        Detection radius
    L : float
        Area side length
    
    Returns
    -------
    reduction : float
        AoI reduction from adding k-th volunteer
    """
    if k <= 0:
        return 0.0
    aoi_before = expected_aoi_from_k(k - 1, R, L)
    aoi_after = expected_aoi_from_k(k, R, L)
    
    if np.isinf(aoi_before):
        return np.inf
    return aoi_before - aoi_after
