"""
config.py - Configuration management for AoI crowd-finding simulation

CORRECTION APPLIED: Adaptive cost ranges based on model thresholds (Bρ, NBρ)
to ensure experiments cover informative parameter regimes.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class PhysicalParams:
    """Physical environment parameters."""
    L: float = 100.0              # Area side length [meters]
    R: float = 10.0               # Detection radius [meters]
    
    @property
    def rho(self) -> float:
        """Coverage ratio (single volunteer)."""
        return np.pi * self.R**2 / self.L**2
    
    def __post_init__(self):
        assert self.L > 0, "Area side must be positive"
        assert 0 < self.R < self.L, "Radius must be in (0, L)"
        assert 2 * self.R < self.L, "Need 2R < L for interior target sampling"


@dataclass
class GameParams:
    """Game theory parameters."""
    N: int = 100                  # Number of volunteers
    B: float = 10.0               # Benefit scaling factor
    c: float = 0.1                # Participation cost (default lowered for baseline)
    
    def __post_init__(self):
        assert self.N > 0, "Must have positive volunteers"
        assert self.B > 0, "Benefit must be positive"
        assert self.c > 0, "Cost must be positive"


@dataclass
class SimulationParams:
    """Simulation execution parameters."""
    T: int = 10000                # Time slots per run
    n_runs: int = 1000            # Monte Carlo runs
    seed: Optional[int] = 42      # Random seed (None for random)
    interior_target: bool = True  # Sample target from interior [R, L-R]²
    
    def __post_init__(self):
        assert self.T > 0, "Must have positive time slots"
        assert self.n_runs > 0, "Must have positive runs"


@dataclass
class SimConfig:
    """Complete simulation configuration."""
    physical: PhysicalParams = field(default_factory=PhysicalParams)
    game: GameParams = field(default_factory=GameParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    
    @property
    def L(self) -> float:
        return self.physical.L
    
    @property
    def R(self) -> float:
        return self.physical.R
    
    @property
    def rho(self) -> float:
        return self.physical.rho
    
    @property
    def N(self) -> int:
        return self.game.N
    
    @property
    def B(self) -> float:
        return self.game.B
    
    @property
    def c(self) -> float:
        return self.game.c
    
    @property
    def B_rho(self) -> float:
        """Threshold for k* > 0: need c < B*rho."""
        return self.B * self.rho
    
    @property
    def NB_rho(self) -> float:
        """Threshold for k_opt > 0: need c < N*B*rho."""
        return self.N * self.B * self.rho


def compute_informative_cost_range(
    N: int,
    R: float,
    L: float,
    B: float,
    for_ne: bool = True,
    n_points: int = 50,
    margin_factor: float = 2.0
) -> np.ndarray:
    """
    Compute cost range that produces informative results.
    
    CORRECTION: Instead of arbitrary [0.1, 10], sweep costs around
    the actual thresholds Bρ (for NE) or NBρ (for social optimum).
    
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
    for_ne : bool
        If True, range for Nash equilibrium (around Bρ)
        If False, range for social optimum (around NBρ)
    n_points : int
        Number of points in range
    margin_factor : float
        How many multiples of threshold to include
    
    Returns
    -------
    c_values : ndarray
        Array of cost values
    """
    rho = np.pi * R**2 / L**2
    
    if for_ne:
        threshold = B * rho  # c_max for k* > 0
    else:
        threshold = N * B * rho  # c_max for k_opt > 0
    
    # Sweep from small positive to margin_factor * threshold
    c_min = threshold * 0.01
    c_max = threshold * margin_factor
    
    return np.linspace(c_min, c_max, n_points)


def compute_normalized_cost_range(
    N: int,
    R: float,
    L: float,
    B: float,
    c_normalized_range: Tuple[float, float] = (0.01, 2.0),
    n_points: int = 50,
    normalize_by: str = 'B_rho'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cost range in normalized units c/(Bρ) or c/(NBρ).
    
    Using normalized cost makes plots comparable across different
    parameter values.
    
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
    c_normalized_range : tuple
        (min, max) in normalized units
    n_points : int
        Number of points
    normalize_by : str
        'B_rho' or 'NB_rho'
    
    Returns
    -------
    c_values : ndarray
        Actual cost values
    c_normalized : ndarray
        Normalized cost values (for plotting)
    """
    rho = np.pi * R**2 / L**2
    
    if normalize_by == 'B_rho':
        scale = B * rho
    elif normalize_by == 'NB_rho':
        scale = N * B * rho
    else:
        raise ValueError(f"Unknown normalize_by: {normalize_by}")
    
    c_normalized = np.linspace(c_normalized_range[0], c_normalized_range[1], n_points)
    c_values = c_normalized * scale
    
    return c_values, c_normalized


@dataclass
class ExperimentGrid:
    """
    Parameter grid for experiments.
    
    CORRECTION: Cost values computed adaptively based on thresholds.
    """
    N_values: List[int] = field(default_factory=lambda: [50, 100, 200])
    R_L_ratios: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])
    B: float = 10.0
    L: float = 100.0
    n_runs: int = 1000
    n_cost_points: int = 50
    
    def get_cost_range_for_ne(self, N: int, R: float) -> np.ndarray:
        """Get cost range for Nash equilibrium analysis."""
        return compute_informative_cost_range(
            N, R, self.L, self.B, for_ne=True,
            n_points=self.n_cost_points, margin_factor=2.0
        )
    
    def get_cost_range_for_opt(self, N: int, R: float) -> np.ndarray:
        """Get cost range for social optimum analysis."""
        return compute_informative_cost_range(
            N, R, self.L, self.B, for_ne=False,
            n_points=self.n_cost_points, margin_factor=2.0
        )
    
    def get_normalized_cost_range(
        self, N: int, R: float,
        c_range: Tuple[float, float] = (0.01, 2.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get cost range in normalized units c/(Bρ)."""
        return compute_normalized_cost_range(
            N, R, self.L, self.B,
            c_normalized_range=c_range,
            n_points=self.n_cost_points,
            normalize_by='B_rho'
        )
    
    def get_R_values(self) -> List[float]:
        """Convert R/L ratios to R values."""
        return [ratio * self.L for ratio in self.R_L_ratios]


# Default configurations
DEFAULT_PHYSICAL = PhysicalParams(L=100.0, R=10.0)
DEFAULT_GAME = GameParams(N=100, B=10.0, c=0.1)
DEFAULT_SIMULATION = SimulationParams(T=10000, n_runs=1000, seed=42, interior_target=True)
DEFAULT_CONFIG = SimConfig(DEFAULT_PHYSICAL, DEFAULT_GAME, DEFAULT_SIMULATION)

EXPERIMENT_GRID = ExperimentGrid()


def print_thresholds(config: SimConfig):
    """Print key thresholds for debugging."""
    print(f"=== Model Thresholds ===")
    print(f"ρ = πR²/L² = {config.rho:.6f}")
    print(f"Bρ = {config.B_rho:.4f}  (max c for k* > 0)")
    print(f"NBρ = {config.NB_rho:.4f}  (max c for k_opt > 0)")
    print(f"Current c = {config.c:.4f}")
    print(f"c < Bρ? {config.c < config.B_rho}  (NE will be non-zero)")
    print(f"c < NBρ? {config.c < config.NB_rho}  (Optimum will be non-zero)")


def suggest_parameters(N: int = 100, R: float = 10.0, L: float = 100.0, B: float = 10.0):
    """Suggest appropriate cost ranges for given parameters."""
    rho = np.pi * R**2 / L**2
    B_rho = B * rho
    NB_rho = N * B * rho
    
    print(f"=== Suggested Parameters ===")
    print(f"Given: N={N}, R={R}, L={L}, B={B}")
    print(f"ρ = {rho:.6f}")
    print(f"")
    print(f"For Nash Equilibrium analysis:")
    print(f"  Threshold: Bρ = {B_rho:.4f}")
    print(f"  Suggested cost range: c ∈ [{B_rho*0.01:.4f}, {B_rho*2:.4f}]")
    print(f"")
    print(f"For Social Optimum analysis:")
    print(f"  Threshold: NBρ = {NB_rho:.4f}")
    print(f"  Suggested cost range: c ∈ [{NB_rho*0.01:.4f}, {NB_rho*2:.4f}]")
    print(f"")
    print(f"For PoA analysis (need k* > 0):")
    print(f"  Restrict to: c < {B_rho:.4f}")
