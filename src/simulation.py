"""
simulation.py - Monte Carlo simulation engine

CORRECTION APPLIED: Target sampled from interior [R, L-R]² to match
analytical theory (no boundary truncation effects).
"""
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Dict, Optional
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
    
    CORRECTION: Target is sampled from interior [R, L-R]² to eliminate
    boundary effects and match analytical formula P_det(k) = 1 - (1-ρ)^k.
    """
    
    def __init__(
        self,
        L: float,
        R: float,
        T: int = 10000,
        seed: Optional[int] = None,
        interior_target: bool = True
    ):
        """
        Initialize simulation.
        
        Parameters
        ----------
        L : float
            Area side length
        R : float
            Detection radius
        T : int
            Number of time slots
        seed : int, optional
            Random seed
        interior_target : bool
            If True, sample target from [R, L-R]² (recommended for theory match)
        """
        self.L = L
        self.R = R
        self.T = T
        self.seed = seed
        self.interior_target = interior_target
        self.rng = np.random.default_rng(seed)
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
        
        CORRECTION: Samples from interior [R, L-R]² when interior_target=True.
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
        L: float,
        R: float,
        T: int = 10000,
        base_seed: int = 42,
        interior_target: bool = True
    ):
        """
        Initialize Monte Carlo runner.
        
        Parameters
        ----------
        L : float
            Area side length
        R : float
            Detection radius
        T : int
            Time slots per run
        base_seed : int
            Base random seed
        interior_target : bool
            If True, sample target from interior (recommended)
        """
        self.L = L
        self.R = R
        self.T = T
        self.base_seed = base_seed
        self.interior_target = interior_target
    
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
    L: float,
    R: float,
    k_values: List[int],
    T: int = 10000,
    n_runs: int = 1000,
    tolerance: float = 0.05,
    interior_target: bool = True
) -> Dict:
    """
    Validate analytical formulas against simulation.
    
    CORRECTION: Uses interior target sampling by default for proper validation.
    
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
    
    return validation
