"""Monte Carlo simulation engine.

Runs single-trajectory and Monte Carlo simulations for AoI dynamics and provides validation helpers against analytical expressions.

Notes:
    Target sampling can be restricted to the interior [R, L-R]^2 to match the constant-coverage assumption used in the analytical model.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    mean_aoi: float
    peak_aoi: int
    detection_rate: float
    n_detections: int
    trajectory: Optional[NDArray[np.int_]] = None


@dataclass
class MonteCarloResult:
    """Aggregated results from Monte Carlo simulation."""
    mean_aoi: float
    mean_aoi_std: float
    mean_peak_aoi: float
    mean_detection_rate: float
    n_runs: int
    individual_results: Optional[List[SimulationResult]] = None


class Simulation:
    """
    Single-run simulation of AoI in crowd-finding scenario.

    Notes:
        When interior_target=True, the target is sampled from the interior
        [R, L-R]^2 to reduce boundary effects. This matches the analytical
        assumption behind P_det(k) = 1 - (1 - ρ)^k with ρ = πR^2 / L^2.
    """
    def __init__(
        self,
        cfg_or_L=None,
        R: Optional[float] = None,
        T: Optional[int] = None,
        seed: Optional[int] = None,
        interior_target: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize a single-run AoI simulation.

        Two calling conventions are supported:

        1) Simulation(cfg: SimConfig)
           Parameters are taken from cfg.physical and cfg.simulation.

        2) Simulation(L: float, R: float, T: int=..., seed: int|None=..., interior_target: bool=...)
           Parameters are passed explicitly.

        Keyword compatibility:
            Simulation(L=..., R=..., ...)
        """
        # Backward-compatible keyword support: Simulation(L=..., R=..., ...)
        if cfg_or_L is None and 'L' in kwargs:
            cfg_or_L = kwargs.pop('L')
        if R is None and 'R' in kwargs:
            R = kwargs.pop('R')
        if kwargs:
            unexpected = ', '.join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        # Lazy import to avoid circular import hazards at module import time.
        try:
            from .config import SimConfig  # type: ignore
        except Exception:  # pragma: no cover
            SimConfig = None  # type: ignore

        if SimConfig is not None and isinstance(cfg_or_L, SimConfig):
            cfg = cfg_or_L
            self.L = float(cfg.physical.L)
            self.R = float(cfg.physical.R)
            self.T = int(cfg.simulation.T if T is None else T)
            self.seed = cfg.simulation.seed if seed is None else seed
            self.interior_target = bool(
                cfg.simulation.interior_target if interior_target is None else interior_target
            )
        else:
            if cfg_or_L is None or R is None:
                raise TypeError(
                    "Simulation(L, R, ...) requires both L and R when the first argument is not a SimConfig"
                )
            self.L = float(cfg_or_L)
            self.R = float(R)
            self.T = int(10000 if T is None else T)
            self.seed = seed
            self.interior_target = bool(True if interior_target is None else interior_target)

        self.rng = np.random.default_rng(self.seed)
        self.reset()
    def reset(self, seed: Optional[int] = None):
        """Reset simulation state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        self.aoi = 0
        self.trajectory = []
        self.detection_times = []
    
    def _sample_target(self) -> NDArray[np.float64]:
        """
        Sample target position.

        When interior_target=True and feasible (R > 0 and 2R < L), samples from
        [R, L-R]^2; otherwise samples from [0, L]^2.
        """
        if self.interior_target and self.R > 0 and 2 * self.R < self.L:
            return self.rng.uniform(self.R, self.L - self.R, size=2)
        else:
            return self.rng.uniform(0, self.L, size=2)
    
    def step(self, k: int) -> Dict:
        """
        Execute one time slot.
        
        Parameters
        ----------
        k : int
            Number of active volunteers this slot
        
        Returns
        -------
        info : dict
            Step information
        """
        detected = False
        
        if k > 0:
            # Volunteers always uniform in full domain [0, L]²
            positions = self.rng.uniform(0, self.L, size=(k, 2))
            
            # Target in interior (correction) or full domain
            target = self._sample_target()
            
            # Check for detection
            distances = np.linalg.norm(positions - target, axis=1)
            detected = np.any(distances <= self.R)
        
        # Update AoI
        if detected:
            self.aoi = 0
            self.detection_times.append(self.t)
        else:
            self.aoi += 1
        
        self.trajectory.append(self.aoi)
        self.t += 1
        
        return {
            't': self.t,
            'aoi': self.aoi,
            'detected': detected,
        }
    
    def run(self, k: int, store_trajectory: bool = False) -> SimulationResult:
        """
        Run complete simulation.
        
        Parameters
        ----------
        k : int
            Number of active volunteers (constant)
        store_trajectory : bool
            Whether to store full trajectory
        
        Returns
        -------
        result : SimulationResult
        """
        self.reset()
        
        for _ in range(self.T):
            self.step(k)
        
        trajectory = np.array(self.trajectory)
        
        return SimulationResult(
            mean_aoi=float(np.mean(trajectory)),
            peak_aoi=int(np.max(trajectory)),
            detection_rate=len(self.detection_times) / self.T,
            n_detections=len(self.detection_times),
            trajectory=trajectory if store_trajectory else None,
        )


class MonteCarloSimulation:
    """
    Monte Carlo simulation runner.
    """
    def __init__(
        self,
        cfg_or_L=None,
        R: Optional[float] = None,
        T: Optional[int] = None,
        base_seed: Optional[int] = None,
        interior_target: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize Monte Carlo runner.

        Supports two calling conventions:

        1) MonteCarloSimulation(cfg: SimConfig)
        2) MonteCarloSimulation(L: float, R: float, T: int=..., base_seed: int=..., interior_target: bool=...)

        Keyword compatibility:
            MonteCarloSimulation(L=..., R=..., ...)
        """
        # Backward-compatible keyword support: MonteCarloSimulation(L=..., R=..., ...)
        if cfg_or_L is None and 'L' in kwargs:
            cfg_or_L = kwargs.pop('L')
        if R is None and 'R' in kwargs:
            R = kwargs.pop('R')
        if kwargs:
            unexpected = ', '.join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        try:
            from .config import SimConfig  # type: ignore
        except Exception:  # pragma: no cover
            SimConfig = None  # type: ignore

        if SimConfig is not None and isinstance(cfg_or_L, SimConfig):
            cfg = cfg_or_L
            self.L = float(cfg.physical.L)
            self.R = float(cfg.physical.R)
            self.T = int(cfg.simulation.T if T is None else T)
            # If cfg.simulation.seed is None, fall back to 0 for deterministic seeding.
            cfg_seed = 0 if cfg.simulation.seed is None else int(cfg.simulation.seed)
            self.base_seed = cfg_seed if base_seed is None else int(base_seed)
            self.interior_target = bool(
                cfg.simulation.interior_target if interior_target is None else interior_target
            )
        else:
            if cfg_or_L is None or R is None:
                raise TypeError(
                    "MonteCarloSimulation(L, R, ...) requires both L and R when the first argument is not a SimConfig"
                )
            self.L = float(cfg_or_L)
            self.R = float(R)
            self.T = int(10000 if T is None else T)
            self.base_seed = 42 if base_seed is None else int(base_seed)
            self.interior_target = bool(True if interior_target is None else interior_target)
    def run(
        self,
        k: int,
        n_runs: int = 1000,
        show_progress: bool = True,
        store_individual: bool = False
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Parameters
        ----------
        k : int
            Number of active volunteers
        n_runs : int
            Number of runs
        show_progress : bool
            Show progress bar
        store_individual : bool
            Store individual run results
        
        Returns
        -------
        result : MonteCarloResult
        """
        mean_aois = []
        peak_aois = []
        detection_rates = []
        individual_results = [] if store_individual else None
        
        iterator = range(n_runs)
        if show_progress:
            iterator = tqdm(iterator, desc=f"MC simulation (k={k})")
        
        for i in iterator:
            sim = Simulation(
                L=self.L,
                R=self.R,
                T=self.T,
                seed=self.base_seed + i,
                interior_target=self.interior_target
            )
            result = sim.run(k, store_trajectory=store_individual)
            
            mean_aois.append(result.mean_aoi)
            peak_aois.append(result.peak_aoi)
            detection_rates.append(result.detection_rate)
            
            if store_individual:
                individual_results.append(result)
        
        return MonteCarloResult(
            mean_aoi=float(np.mean(mean_aois)),
            mean_aoi_std=float(np.std(mean_aois) / np.sqrt(n_runs)),  # Standard error
            mean_peak_aoi=float(np.mean(peak_aois)),
            mean_detection_rate=float(np.mean(detection_rates)),
            n_runs=n_runs,
            individual_results=individual_results,
        )


def run_parameter_sweep(
    L: float,
    R: float,
    k_values: List[int],
    T: int = 10000,
    n_runs: int = 100,
    show_progress: bool = True,
    interior_target: bool = True
) -> Dict[str, NDArray]:
    """
    Run simulations across range of k values.
    
    Parameters
    ----------
    L : float
        Area side length
    R : float
        Detection radius
    k_values : list of int
        Values of k to simulate
    T : int
        Time slots per run
    n_runs : int
        Runs per k value
    show_progress : bool
        Show progress
    interior_target : bool
        Use interior target sampling
    
    Returns
    -------
    results : dict
        Arrays of results indexed by k
    """
    n = len(k_values)
    results = {
        'k': np.array(k_values),
        'mean_aoi': np.zeros(n),
        'mean_aoi_std': np.zeros(n),
        'mean_peak_aoi': np.zeros(n),
        'detection_rate': np.zeros(n),
    }
    
    mc = MonteCarloSimulation(
        L=L, R=R, T=T, interior_target=interior_target
    )
    
    for i, k in enumerate(k_values):
        if show_progress:
            print(f"Running k={k} ({i+1}/{n})")
        
        mc_result = mc.run(k, n_runs=n_runs, show_progress=False)
        
        results['mean_aoi'][i] = mc_result.mean_aoi
        results['mean_aoi_std'][i] = mc_result.mean_aoi_std
        results['mean_peak_aoi'][i] = mc_result.mean_peak_aoi
        results['detection_rate'][i] = mc_result.mean_detection_rate
    
    return results


def validate_analytical(
    cfg_or_L,
    R: Optional[float] = None,
    k_values: List[int] = None,
    T: int = 10000,
    n_runs: int = 1000,
    tolerance: float = 0.05,
    interior_target: bool = True
) -> Dict:
    """
    Validate analytical formulas against simulation.

    Notes:
        interior_target defaults to True to align the simulation setup with the
        assumptions used by the closed-form coverage probability.
    
    Parameters
    ----------
    L : float
        Area side length
    R : float
        Detection radius
    k_values : list of int
        Values of k to test
    T : int
        Time slots per run
    n_runs : int
        Runs per k value
    tolerance : float
        Relative tolerance for validation
    interior_target : bool
        Use interior target sampling
    
    Returns
    -------
    validation : dict
        Validation results
    """
    if k_values is None:
        raise TypeError("validate_analytical(...) requires k_values")

    try:
        from .config import SimConfig  # type: ignore
    except Exception:  # pragma: no cover
        SimConfig = None  # type: ignore

    if SimConfig is not None and isinstance(cfg_or_L, SimConfig):
        cfg = cfg_or_L
        L = float(cfg.physical.L)
        R_val = float(cfg.physical.R)
        if T == 10000:
            T = int(cfg.simulation.T)
        if interior_target is True:
            interior_target = bool(cfg.simulation.interior_target)
        R = R_val
    else:
        L = float(cfg_or_L)
        if R is None:
            raise TypeError("validate_analytical(L, R, ...) requires R when the first argument is not a SimConfig")

    from .aoi import expected_aoi_from_k
    
    results = run_parameter_sweep(
        L=L, R=R, k_values=k_values, T=T, n_runs=n_runs,
        show_progress=True, interior_target=interior_target
    )
    
    validation = {
        'k': np.array(k_values),
        'analytical_aoi': np.zeros(len(k_values)),
        'simulated_aoi': results['mean_aoi'],
        'simulated_aoi_std': results['mean_aoi_std'],
        'relative_error': np.zeros(len(k_values)),
        'within_tolerance': np.zeros(len(k_values), dtype=bool),
        'within_3sigma': np.zeros(len(k_values), dtype=bool),
    }
    
    for i, k in enumerate(k_values):
        analytical = expected_aoi_from_k(k, R, L)
        validation['analytical_aoi'][i] = analytical
        
        if analytical > 0 and not np.isinf(analytical):
            error = abs(results['mean_aoi'][i] - analytical) / analytical
            # Check if within 3 standard errors (Monte Carlo error)
            within_3sigma = abs(results['mean_aoi'][i] - analytical) < 3 * results['mean_aoi_std'][i]
        else:
            error = 0.0 if results['mean_aoi'][i] == analytical else np.inf
            within_3sigma = True
        
        validation['relative_error'][i] = error
        validation['within_tolerance'][i] = error < tolerance
        validation['within_3sigma'][i] = within_3sigma
    
    validation['all_within_tolerance'] = np.all(validation['within_tolerance'])
    validation['all_within_3sigma'] = np.all(validation['within_3sigma'])
    # Backward-compatible aliases used by experiments/tests
    validation['passed'] = validation['within_tolerance']
    validation['all_passed'] = bool(validation['all_within_tolerance'])
    
    return validation


# =============================================================================
# HETEROGENEOUS COST VALIDATION (Version 2.0)
# =============================================================================

from .config import HeterogeneousCostParams


def simulate_threshold_equilibrium(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams,
    n_runs: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Validate threshold equilibrium via Monte Carlo.
    
    Each run:
    1. Sample costs c_i ~ F
    2. Compute analytical threshold c̄*
    3. Count volunteers with c_i <= c̄*
    4. Compare to predicted k*
    
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
    n_runs : int
        Number of Monte Carlo runs
    seed : int
        Random seed
    
    Returns
    -------
    results : dict
        Validation results
    """
    from .game import find_nash_heterogeneous, find_nash_heterogeneous_with_costs
    
    rng = np.random.default_rng(seed)
    
    # Analytical prediction
    k_star_analytical, c_bar_star = find_nash_heterogeneous(N, rho, B, cost_params)
    
    # Monte Carlo
    k_star_samples = []
    threshold_samples = []
    
    for i in range(n_runs):
        costs = cost_params.sample(N, rng)
        k_star_sim, c_bar_sim, _ = find_nash_heterogeneous_with_costs(N, rho, B, costs)
        k_star_samples.append(k_star_sim)
        threshold_samples.append(c_bar_sim)
    
    k_star_samples = np.array(k_star_samples)
    threshold_samples = np.array(threshold_samples)
    
    return {
        'n_runs': n_runs,
        'N': N,
        'rho': rho,
        'B': B,
        'c_min': cost_params.c_min,
        'c_max': cost_params.c_max,
        'distribution': cost_params.distribution,
        # Analytical
        'k_star_analytical': k_star_analytical,
        'c_bar_star_analytical': c_bar_star,
        # Simulated
        'k_star_mean': float(np.mean(k_star_samples)),
        'k_star_std': float(np.std(k_star_samples)),
        'k_star_min': int(np.min(k_star_samples)),
        'k_star_max': int(np.max(k_star_samples)),
        'c_bar_star_mean': float(np.mean(threshold_samples)),
        # Comparison
        'k_star_error': float(np.mean(k_star_samples) - k_star_analytical),
        'k_star_relative_error': float(abs(np.mean(k_star_samples) - k_star_analytical) / max(1, k_star_analytical)),
    }


def simulate_social_optimum_heterogeneous(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams,
    n_runs: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Validate social optimum selection via Monte Carlo.
    
    Each run:
    1. Sample costs
    2. Find optimal selection (greedy by cost)
    3. Compare k_opt to analytical prediction
    
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
    n_runs : int
        Number of Monte Carlo runs
    seed : int
        Random seed
    
    Returns
    -------
    results : dict
        Validation results
    """
    from .game import find_social_optimum_heterogeneous, find_social_optimum_heterogeneous_expected
    
    rng = np.random.default_rng(seed)
    
    # Analytical prediction
    k_opt_analytical, c_bar_opt = find_social_optimum_heterogeneous_expected(N, rho, B, cost_params)
    
    # Monte Carlo
    k_opt_samples = []
    welfare_samples = []
    
    for i in range(n_runs):
        costs = cost_params.sample(N, rng)
        k_opt_sim, _, welfare_sim = find_social_optimum_heterogeneous(N, rho, B, costs)
        k_opt_samples.append(k_opt_sim)
        welfare_samples.append(welfare_sim)
    
    k_opt_samples = np.array(k_opt_samples)
    welfare_samples = np.array(welfare_samples)
    
    return {
        'n_runs': n_runs,
        'N': N,
        'rho': rho,
        'B': B,
        'c_min': cost_params.c_min,
        'c_max': cost_params.c_max,
        'distribution': cost_params.distribution,
        # Analytical
        'k_opt_analytical': k_opt_analytical,
        'c_bar_opt_analytical': c_bar_opt,
        # Simulated
        'k_opt_mean': float(np.mean(k_opt_samples)),
        'k_opt_std': float(np.std(k_opt_samples)),
        'k_opt_min': int(np.min(k_opt_samples)),
        'k_opt_max': int(np.max(k_opt_samples)),
        'welfare_mean': float(np.mean(welfare_samples)),
        'welfare_std': float(np.std(welfare_samples)),
        # Comparison
        'k_opt_error': float(np.mean(k_opt_samples) - k_opt_analytical),
        'k_opt_relative_error': float(abs(np.mean(k_opt_samples) - k_opt_analytical) / max(1, k_opt_analytical)),
    }


def simulate_poa_heterogeneous(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams,
    n_runs: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Validate Price of Anarchy via Monte Carlo.
    
    Each run:
    1. Sample costs
    2. Find Nash equilibrium welfare
    3. Find optimal welfare
    4. Compute PoA
    
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
    n_runs : int
        Number of Monte Carlo runs
    seed : int
        Random seed
    
    Returns
    -------
    results : dict
        Validation results
    """
    from .game import (
        find_nash_heterogeneous_with_costs,
        find_social_optimum_heterogeneous,
        social_welfare_heterogeneous,
        price_of_anarchy_heterogeneous
    )
    
    rng = np.random.default_rng(seed)
    
    # Analytical prediction
    poa_analytical = price_of_anarchy_heterogeneous(N, rho, B, cost_params)
    
    # Monte Carlo
    poa_samples = []
    welfare_nash_samples = []
    welfare_opt_samples = []
    
    for i in range(n_runs):
        costs = cost_params.sample(N, rng)
        
        # Nash equilibrium
        k_nash, c_bar_nash, active_nash = find_nash_heterogeneous_with_costs(N, rho, B, costs)
        welfare_nash = social_welfare_heterogeneous(k_nash, N, rho, B, np.sum(costs[active_nash]))
        
        # Social optimum
        k_opt, active_opt, welfare_opt = find_social_optimum_heterogeneous(N, rho, B, costs)
        
        if welfare_nash > 0:
            poa = welfare_opt / welfare_nash
        else:
            poa = np.inf
        
        poa_samples.append(poa)
        welfare_nash_samples.append(welfare_nash)
        welfare_opt_samples.append(welfare_opt)
    
    poa_samples = np.array(poa_samples)
    poa_finite = poa_samples[np.isfinite(poa_samples)]
    
    return {
        'n_runs': n_runs,
        'N': N,
        'rho': rho,
        'B': B,
        'c_min': cost_params.c_min,
        'c_max': cost_params.c_max,
        # Analytical
        'poa_analytical': poa_analytical,
        # Simulated
        'poa_mean': float(np.mean(poa_finite)) if len(poa_finite) > 0 else np.inf,
        'poa_std': float(np.std(poa_finite)) if len(poa_finite) > 0 else np.inf,
        'poa_median': float(np.median(poa_finite)) if len(poa_finite) > 0 else np.inf,
        'poa_max': float(np.max(poa_finite)) if len(poa_finite) > 0 else np.inf,
        'n_infinite_poa': int(np.sum(~np.isfinite(poa_samples))),
        'welfare_nash_mean': float(np.mean(welfare_nash_samples)),
        'welfare_opt_mean': float(np.mean(welfare_opt_samples)),
    }


def validate_heterogeneous_model(
    N: int,
    rho: float,
    B: float,
    cost_params: HeterogeneousCostParams,
    n_runs: int = 1000,
    seed: int = 42,
    tolerance: float = 0.10
) -> Dict:
    """
    Complete validation suite for heterogeneous model.
    
    Validates:
    1. Nash equilibrium (threshold structure)
    2. Social optimum (selection rule)
    3. Price of Anarchy
    
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
    n_runs : int
        Number of Monte Carlo runs
    seed : int
        Random seed
    tolerance : float
        Relative tolerance for validation
    
    Returns
    -------
    results : dict
        Comprehensive validation results
    """
    # Run all validations
    nash_validation = simulate_threshold_equilibrium(N, rho, B, cost_params, n_runs, seed)
    opt_validation = simulate_social_optimum_heterogeneous(N, rho, B, cost_params, n_runs, seed)
    poa_validation = simulate_poa_heterogeneous(N, rho, B, cost_params, n_runs, seed)
    
    # Check if within tolerance
    nash_ok = nash_validation['k_star_relative_error'] < tolerance
    opt_ok = opt_validation['k_opt_relative_error'] < tolerance
    
    return {
        'nash': nash_validation,
        'optimum': opt_validation,
        'poa': poa_validation,
        'nash_validated': nash_ok,
        'optimum_validated': opt_ok,
        'all_validated': nash_ok and opt_ok,
        'tolerance': tolerance,
    }


def sweep_heterogeneity_validation(
    N: int,
    rho: float,
    B: float,
    mean_cost: float,
    spread_ratios: List[float],
    n_runs: int = 500,
    seed: int = 42
) -> Dict[str, NDArray]:
    """
    Validate model across heterogeneity levels.
    
    Parameters
    ----------
    N : int
        Number of volunteers
    rho : float
        Coverage ratio
    B : float
        Benefit parameter
    mean_cost : float
        Fixed mean cost
    spread_ratios : list of float
        c_max/c_min ratios to test
    n_runs : int
        Monte Carlo runs per configuration
    seed : int
        Random seed
    
    Returns
    -------
    results : dict of arrays
    """
    n = len(spread_ratios)
    results = {
        'spread_ratio': np.array(spread_ratios),
        'c_min': np.zeros(n),
        'c_max': np.zeros(n),
        'k_star_analytical': np.zeros(n, dtype=int),
        'k_star_simulated': np.zeros(n),
        'k_star_error': np.zeros(n),
        'k_opt_analytical': np.zeros(n, dtype=int),
        'k_opt_simulated': np.zeros(n),
        'k_opt_error': np.zeros(n),
        'poa_analytical': np.zeros(n),
        'poa_simulated': np.zeros(n),
    }
    
    for i, ratio in enumerate(spread_ratios):
        c_min = 2 * mean_cost / (1 + ratio)
        c_max = ratio * c_min
        cost_params = HeterogeneousCostParams(c_min=c_min, c_max=c_max)
        
        validation = validate_heterogeneous_model(N, rho, B, cost_params, n_runs, seed + i * 1000)
        
        results['c_min'][i] = c_min
        results['c_max'][i] = c_max
        results['k_star_analytical'][i] = validation['nash']['k_star_analytical']
        results['k_star_simulated'][i] = validation['nash']['k_star_mean']
        results['k_star_error'][i] = validation['nash']['k_star_relative_error']
        results['k_opt_analytical'][i] = validation['optimum']['k_opt_analytical']
        results['k_opt_simulated'][i] = validation['optimum']['k_opt_mean']
        results['k_opt_error'][i] = validation['optimum']['k_opt_relative_error']
        results['poa_analytical'][i] = validation['poa']['poa_analytical']
        results['poa_simulated'][i] = validation['poa']['poa_mean']
    
    return results