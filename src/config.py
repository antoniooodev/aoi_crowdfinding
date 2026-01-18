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


# =============================================================================
# HETEROGENEOUS COST MODEL EXTENSIONS (Version 2.0)
# =============================================================================

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def uniform_cdf(c: float, c_min: float, c_max: float) -> float:
    """CDF of uniform distribution on [c_min, c_max]."""
    if c <= c_min:
        return 0.0
    if c >= c_max:
        return 1.0
    return (c - c_min) / (c_max - c_min)


def uniform_pdf(c: float, c_min: float, c_max: float) -> float:
    """PDF of uniform distribution on [c_min, c_max]."""
    if c < c_min or c > c_max:
        return 0.0
    return 1.0 / (c_max - c_min)


def uniform_quantile(q: float, c_min: float, c_max: float) -> float:
    """Quantile function (inverse CDF) of uniform distribution."""
    return c_min + q * (c_max - c_min)


def truncated_normal_cdf(c: float, c_min: float, c_max: float, 
                         mu: float, sigma: float) -> float:
    """CDF of truncated normal distribution on [c_min, c_max]."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for truncated normal distribution")
    if c <= c_min:
        return 0.0
    if c >= c_max:
        return 1.0
    
    a = (c_min - mu) / sigma
    b = (c_max - mu) / sigma
    z = (c - mu) / sigma
    
    Phi_a = stats.norm.cdf(a)
    Phi_b = stats.norm.cdf(b)
    Phi_z = stats.norm.cdf(z)
    
    return (Phi_z - Phi_a) / (Phi_b - Phi_a)


def truncated_normal_quantile(q: float, c_min: float, c_max: float,
                              mu: float, sigma: float) -> float:
    """Quantile function of truncated normal distribution."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for truncated normal distribution")
    
    a = (c_min - mu) / sigma
    b = (c_max - mu) / sigma
    
    Phi_a = stats.norm.cdf(a)
    Phi_b = stats.norm.cdf(b)
    
    target_Phi = Phi_a + q * (Phi_b - Phi_a)
    z = stats.norm.ppf(target_Phi)
    
    return mu + sigma * z


def truncated_exponential_cdf(c: float, c_min: float, c_max: float,
                              lambda_exp: float) -> float:
    """CDF of truncated exponential distribution on [c_min, c_max]."""
    if c <= c_min:
        return 0.0
    if c >= c_max:
        return 1.0
    
    numerator = 1 - np.exp(-lambda_exp * (c - c_min))
    denominator = 1 - np.exp(-lambda_exp * (c_max - c_min))
    
    return numerator / denominator


def truncated_exponential_quantile(q: float, c_min: float, c_max: float,
                                   lambda_exp: float) -> float:
    """Quantile function of truncated exponential distribution."""
    denom = 1 - np.exp(-lambda_exp * (c_max - c_min))
    return c_min - np.log(1 - q * denom) / lambda_exp


@dataclass
class HeterogeneousCostParams:
    """
    Heterogeneous cost distribution parameters.
    
    Supports: uniform, truncated_normal, truncated_exponential
    
    Parameters
    ----------
    c_min : float
        Minimum cost
    c_max : float
        Maximum cost
    distribution : str
        Distribution type: "uniform", "truncated_normal", "truncated_exponential"
    mu : float, optional
        Mean for truncated normal (defaults to midpoint)
    sigma : float, optional
        Std dev for truncated normal (defaults to (c_max-c_min)/4)
    lambda_exp : float, optional
        Rate for truncated exponential (defaults to 2/(c_max-c_min))
    """
    c_min: float = 0.5
    c_max: float = 2.0
    distribution: str = "uniform"
    mu: Optional[float] = None
    sigma: Optional[float] = None
    lambda_exp: Optional[float] = None
    
    def __post_init__(self):
        assert self.c_min > 0, "Minimum cost must be positive"
        assert self.c_max >= self.c_min, "Maximum cost must be >= minimum"
        assert self.distribution in ["uniform", "truncated_normal", "truncated_exponential"], \
            f"Unknown distribution: {self.distribution}"
        
        # Set defaults for distribution parameters
        if self.distribution == "truncated_normal":
            if self.mu is None:
                self.mu = (self.c_min + self.c_max) / 2
            if self.sigma is None:
                self.sigma = (self.c_max - self.c_min) / 4
        
        if self.distribution == "truncated_exponential":
            if self.lambda_exp is None:
                self.lambda_exp = 2.0 / (self.c_max - self.c_min)
    
    @property
    def mean_cost(self) -> float:
        """Expected cost E[c_i]."""
        if self.distribution == "uniform":
            return (self.c_min + self.c_max) / 2
        elif self.distribution == "truncated_normal":
            from scipy.integrate import quad
            def integrand(c):
                return c * self.pdf(c)
            result, _ = quad(integrand, self.c_min, self.c_max)
            return result
        elif self.distribution == "truncated_exponential":
            from scipy.integrate import quad
            def integrand(c):
                return c * self.pdf(c)
            result, _ = quad(integrand, self.c_min, self.c_max)
            return result
        return (self.c_min + self.c_max) / 2
    
    @property
    def cost_spread(self) -> float:
        """Cost spread Δc = c_max - c_min."""
        return self.c_max - self.c_min
    
    @property
    def is_homogeneous(self) -> bool:
        """True if c_min == c_max (degenerate distribution)."""
        return np.isclose(self.c_min, self.c_max)
    
    @property
    def heterogeneity_ratio(self) -> float:
        """Ratio c_max / c_min, measuring spread."""
        return self.c_max / self.c_min
    
    def cdf(self, c: float) -> float:
        """Cumulative distribution function F(c) = P(c_i <= c)."""
        if self.distribution == "uniform":
            return uniform_cdf(c, self.c_min, self.c_max)
        elif self.distribution == "truncated_normal":
            return truncated_normal_cdf(c, self.c_min, self.c_max, self.mu, self.sigma)
        elif self.distribution == "truncated_exponential":
            return truncated_exponential_cdf(c, self.c_min, self.c_max, self.lambda_exp)
        raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def pdf(self, c: float) -> float:
        """Probability density function f(c)."""
        if c < self.c_min or c > self.c_max:
            return 0.0
        
        if self.distribution == "uniform":
            return uniform_pdf(c, self.c_min, self.c_max)
        elif self.distribution == "truncated_normal":
            a = (self.c_min - self.mu) / self.sigma
            b = (self.c_max - self.mu) / self.sigma
            z = (c - self.mu) / self.sigma
            normalizer = stats.norm.cdf(b) - stats.norm.cdf(a)
            return stats.norm.pdf(z) / (self.sigma * normalizer)
        elif self.distribution == "truncated_exponential":
            normalizer = 1 - np.exp(-self.lambda_exp * (self.c_max - self.c_min))
            return self.lambda_exp * np.exp(-self.lambda_exp * (c - self.c_min)) / normalizer
        raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def quantile(self, q: float) -> float:
        """Quantile function F^{-1}(q)."""
        q = np.clip(q, 0.0, 1.0)
        
        if self.distribution == "uniform":
            return uniform_quantile(q, self.c_min, self.c_max)
        elif self.distribution == "truncated_normal":
            return truncated_normal_quantile(q, self.c_min, self.c_max, self.mu, self.sigma)
        elif self.distribution == "truncated_exponential":
            return truncated_exponential_quantile(q, self.c_min, self.c_max, self.lambda_exp)
        raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample n costs from the distribution."""
        if rng is None:
            rng = np.random.default_rng()
        
        if self.distribution == "uniform":
            return rng.uniform(self.c_min, self.c_max, size=n)
        elif self.distribution == "truncated_normal":
            u = rng.uniform(0, 1, size=n)
            return np.array([self.quantile(ui) for ui in u])
        elif self.distribution == "truncated_exponential":
            u = rng.uniform(0, 1, size=n)
            return np.array([self.quantile(ui) for ui in u])
        raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def is_log_concave(self) -> bool:
        """
        Check if the distribution is log-concave.
        
        Log-concavity guarantees uniqueness of Nash equilibrium.
        """
        return self.distribution in ["uniform", "truncated_normal", "truncated_exponential"]
    
    def expected_sum_of_k_lowest(self, k: int, N: int) -> float:
        """
        Expected sum of k lowest order statistics from N samples.
        
        For uniform: E[sum of k lowest] = k * c_min + k(k+1)/(2(N+1)) * Δc
        """
        if k <= 0:
            return 0.0
        if k > N:
            k = N
        
        if self.distribution == "uniform":
            delta_c = self.c_max - self.c_min
            return k * self.c_min + k * (k + 1) / (2 * (N + 1)) * delta_c
        else:
            # Numerical approximation: E[c_{(j)}] ≈ F^{-1}(j/(N+1))
            total = 0.0
            for j in range(1, k + 1):
                total += self.quantile(j / (N + 1))
            return total
    
    def expected_kth_order_statistic(self, k: int, N: int) -> float:
        """Expected value of k-th order statistic from N samples."""
        if k <= 0 or k > N:
            raise ValueError(f"k must be in [1, N], got k={k}, N={N}")
        
        if self.distribution == "uniform":
            return self.c_min + k / (N + 1) * self.cost_spread
        else:
            return self.quantile(k / (N + 1))
    
    @classmethod
    def from_homogeneous(cls, c: float) -> 'HeterogeneousCostParams':
        """Create a degenerate (homogeneous) distribution with single cost c."""
        return cls(c_min=c, c_max=c, distribution="uniform")


@dataclass
class HeterogeneousSimConfig:
    """Complete simulation configuration with heterogeneous costs."""
    physical: PhysicalParams = field(default_factory=PhysicalParams)
    N: int = 100
    B: float = 10.0
    cost_params: HeterogeneousCostParams = field(default_factory=HeterogeneousCostParams)
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
    def B_rho(self) -> float:
        return self.B * self.rho
    
    @property
    def NB_rho(self) -> float:
        return self.N * self.B * self.rho
    
    @property
    def c_min(self) -> float:
        return self.cost_params.c_min
    
    @property
    def c_max(self) -> float:
        return self.cost_params.c_max


def compute_heterogeneity_sweep(
    mean_cost: float,
    spread_ratios: List[float],
) -> List[HeterogeneousCostParams]:
    """
    Generate cost parameters with varying heterogeneity around fixed mean.
    
    Parameters
    ----------
    mean_cost : float
        Mean cost (fixed)
    spread_ratios : list of float
        List of c_max/c_min ratios to test
    
    Returns
    -------
    params_list : list of HeterogeneousCostParams
    """
    result = []
    for ratio in spread_ratios:
        c_min = 2 * mean_cost / (1 + ratio)
        c_max = ratio * c_min
        result.append(HeterogeneousCostParams(c_min=c_min, c_max=c_max))
    return result


def print_heterogeneous_thresholds(N: int, rho: float, B: float, 
                                    cost_params: HeterogeneousCostParams):
    """Print thresholds for heterogeneous model."""
    print(f"=== Heterogeneous Model Thresholds ===")
    print(f"N = {N}, ρ = {rho:.6f}, B = {B}")
    print(f"Cost distribution: {cost_params.distribution}")
    print(f"c_min = {cost_params.c_min:.4f}, c_max = {cost_params.c_max:.4f}")
    print(f"Mean cost = {cost_params.mean_cost:.4f}")
    print(f"Heterogeneity ratio = {cost_params.heterogeneity_ratio:.2f}")
    print(f"Bρ = {B * rho:.4f}")
    print(f"NBρ = {N * B * rho:.4f}")
    print(f"Log-concave? {cost_params.is_log_concave()}")


# Default heterogeneous configuration
DEFAULT_HETEROGENEOUS_COSTS = HeterogeneousCostParams(c_min=0.5, c_max=2.0, distribution="uniform")