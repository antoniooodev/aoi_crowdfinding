# Simulation Development Pipeline

## AoI-Aware Crowd-Finding for Emergency Search

---

# Table of Contents

1. [Project Structure](#1-project-structure)
2. [Development Environment Setup](#2-development-environment-setup)
3. [Module Specifications](#3-module-specifications)
4. [Implementation Pipeline](#4-implementation-pipeline)
5. [Testing Strategy](#5-testing-strategy)
6. [Experiment Design](#6-experiment-design)
7. [Results Analysis](#7-results-analysis)
8. [Visualization Guidelines](#8-visualization-guidelines)
9. [Execution Checklist](#9-execution-checklist)

---

# 1. Project Structure

## 1.1 Directory Layout

```
aoi_crowdfinding/
│
├── docs/
│   ├── formal_model.md              # Theoretical model (Phase 1-2)
│   └── simulation_pipeline.md       # This document
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration and parameters
│   ├── spatial.py                   # 2D geometry and coverage
│   ├── aoi.py                       # Age of Information computations
│   ├── game.py                      # Game theory solvers
│   ├── simulation.py                # Monte Carlo simulation engine
│   ├── stackelberg.py               # Stackelberg game analysis
│   └── visualization.py             # Plotting utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_spatial.py
│   ├── test_aoi.py
│   ├── test_game.py
│   ├── test_simulation.py
│   └── test_integration.py
│
├── experiments/
│   ├── exp01_equilibrium_analysis.py
│   ├── exp02_social_optimum.py
│   ├── exp03_price_of_anarchy.py
│   ├── exp04_stackelberg.py
│   ├── exp05_sensitivity.py
│   └── run_all.py
│
├── results/
│   ├── data/                        # Raw numerical results
│   │   └── .gitkeep
│   └── figures/                     # Generated plots
│       └── .gitkeep
│
├── notebooks/
│   └── analysis.ipynb               # Interactive analysis
│
├── requirements.txt
├── setup.py
└── README.md
```

## 1.2 File Descriptions

| File | Purpose | Dependencies |
|------|---------|--------------|
| `config.py` | Centralized parameters, dataclasses | None |
| `spatial.py` | Geometry, positions, coverage | numpy |
| `aoi.py` | AoI formulas, trajectories | numpy, spatial |
| `game.py` | NE solver, welfare, PoA | numpy, aoi |
| `simulation.py` | Monte Carlo engine | numpy, spatial, aoi |
| `stackelberg.py` | Incentive optimization | numpy, game |
| `visualization.py` | Publication plots | matplotlib, numpy |

---

# 2. Development Environment Setup

## 2.1 Requirements

```
# requirements.txt
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pandas>=2.0.0
pytest>=7.0.0
pytest-cov>=4.0.0
tqdm>=4.65.0
seaborn>=0.12.0
```

## 2.2 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 2.3 Setup Script

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="aoi_crowdfinding",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    author="Antonio",
    description="AoI-Aware Crowd-Finding Game Theory Simulation",
)
```

---

# 3. Module Specifications

## 3.1 Configuration Module (`config.py`)

### Purpose
Centralized parameter management using dataclasses for type safety and validation.

### Specification

```python
"""
config.py - Configuration management for AoI crowd-finding simulation
"""
from dataclasses import dataclass, field
from typing import List, Optional
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


@dataclass
class GameParams:
    """Game theory parameters."""
    N: int = 100                  # Number of volunteers
    B: float = 10.0               # Benefit scaling factor
    c: float = 1.0                # Participation cost
    
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


@dataclass
class ExperimentGrid:
    """Parameter grid for experiments."""
    N_values: List[int] = field(default_factory=lambda: [50, 100, 200])
    R_values: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])
    c_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0])
    B_values: List[float] = field(default_factory=lambda: [10.0])
    L: float = 100.0
    n_runs: int = 1000


# Default configurations
DEFAULT_CONFIG = SimConfig()

EXPERIMENT_CONFIG = ExperimentGrid()
```

### Interface

| Class | Key Attributes | Key Methods |
|-------|----------------|-------------|
| `PhysicalParams` | L, R | `rho` (property) |
| `GameParams` | N, B, c | — |
| `SimulationParams` | T, n_runs, seed | — |
| `SimConfig` | physical, game, simulation | All properties |
| `ExperimentGrid` | N_values, R_values, c_values | — |

---

## 3.2 Spatial Module (`spatial.py`)

### Purpose
Handle 2D geometry: volunteer positions, target placement, coverage calculations.

### Specification

```python
"""
spatial.py - 2D spatial computations for crowd-finding
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
    Generate n uniformly distributed positions in [0, L]^2.
    
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
    seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Generate single target position uniformly in [0, L]^2.
    
    Parameters
    ----------
    L : float
        Side length of square area
    seed : int, optional
        Random seed
    
    Returns
    -------
    target : ndarray of shape (2,)
        Target (x, y) coordinates
    """
    rng = np.random.default_rng(seed)
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
    
    P_det(k) = 1 - (1 - rho)^k  where rho = pi*R^2/L^2
    
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
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Estimate coverage probability via Monte Carlo simulation.
    
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
        positions = rng.uniform(0, L, size=(k, 2))
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
```

### Interface Summary

| Function | Input | Output |
|----------|-------|--------|
| `generate_positions` | n, L, seed | (n, 2) array |
| `generate_target` | L, seed | (2,) array |
| `compute_distances` | positions, target | (n,) array |
| `coverage_mask` | distances, R | (n,) bool array |
| `count_covering` | positions, target, R | int |
| `analytical_coverage_prob` | k, R, L | float |
| `empirical_coverage_prob` | k, R, L, n_samples | (mean, std_err) |
| `coverage_probability_vectorized` | k_values, R, L | array |

### Test Cases

```python
# test_spatial.py
import numpy as np
import pytest
from src.spatial import *

class TestPositionGeneration:
    def test_shape(self):
        pos = generate_positions(100, 50.0, seed=42)
        assert pos.shape == (100, 2)
    
    def test_bounds(self):
        pos = generate_positions(1000, 50.0, seed=42)
        assert np.all(pos >= 0)
        assert np.all(pos <= 50.0)
    
    def test_reproducibility(self):
        pos1 = generate_positions(10, 50.0, seed=42)
        pos2 = generate_positions(10, 50.0, seed=42)
        np.testing.assert_array_equal(pos1, pos2)


class TestCoverage:
    def test_zero_volunteers(self):
        assert analytical_coverage_prob(0, 10.0, 100.0) == 0.0
    
    def test_monotonicity(self):
        probs = [analytical_coverage_prob(k, 10.0, 100.0) for k in range(101)]
        assert all(probs[i] <= probs[i+1] for i in range(100))
    
    def test_limit(self):
        prob = analytical_coverage_prob(10000, 10.0, 100.0)
        assert prob > 0.999
    
    def test_empirical_matches_analytical(self):
        k, R, L = 50, 10.0, 100.0
        analytical = analytical_coverage_prob(k, R, L)
        empirical, std_err = empirical_coverage_prob(k, R, L, n_samples=50000, seed=42)
        assert abs(empirical - analytical) < 3 * std_err  # 3-sigma


class TestDistances:
    def test_zero_distance(self):
        pos = np.array([[5.0, 5.0]])
        target = np.array([5.0, 5.0])
        dist = compute_distances(pos, target)
        assert dist[0] == 0.0
    
    def test_known_distance(self):
        pos = np.array([[0.0, 0.0]])
        target = np.array([3.0, 4.0])
        dist = compute_distances(pos, target)
        assert dist[0] == 5.0
```

---

## 3.3 AoI Module (`aoi.py`)

### Purpose
Compute Age of Information metrics: expected AoI, trajectories, time-averages.

### Specification

```python
"""
aoi.py - Age of Information computations
"""
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from .spatial import analytical_coverage_prob


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
    rho = np.pi * R**2 / L**2
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
    """
    Fast vectorized AoI trajectory simulation.
    
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
    
    # Generate detection events
    detections = rng.random(T) < P_det
    
    # Find detection times
    detection_times = np.where(detections)[0]
    
    if len(detection_times) == 0:
        # No detections: AoI increases linearly
        return np.arange(T)
    
    # Build trajectory
    trajectory = np.zeros(T, dtype=np.int64)
    
    # Before first detection
    trajectory[:detection_times[0]] = np.arange(detection_times[0])
    
    # Between detections
    for i in range(len(detection_times) - 1):
        start = detection_times[i]
        end = detection_times[i + 1]
        trajectory[start:end] = np.arange(end - start)
    
    # After last detection
    last = detection_times[-1]
    trajectory[last:] = np.arange(T - last)
    
    return trajectory


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


def aoi_percentile(trajectory: NDArray[np.int_], percentile: float) -> float:
    """
    Compute percentile of AoI distribution.
    
    Parameters
    ----------
    trajectory : ndarray
        AoI values over time
    percentile : float
        Percentile (0-100)
    
    Returns
    -------
    value : float
        AoI value at given percentile
    """
    return float(np.percentile(trajectory, percentile))


def detection_times(trajectory: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Extract detection times from trajectory (where AoI resets to 0).
    
    Parameters
    ----------
    trajectory : ndarray
        AoI values over time
    
    Returns
    -------
    times : ndarray
        Indices where detections occurred
    """
    # Detection at t if trajectory[t] == 0 and (t == 0 or trajectory[t-1] > 0)
    resets = np.where(trajectory == 0)[0]
    if len(resets) == 0:
        return np.array([], dtype=np.int64)
    
    # Filter to actual resets (not consecutive zeros)
    mask = np.concatenate([[True], np.diff(resets) > 1])
    return resets[mask]


def inter_detection_times(trajectory: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Compute inter-detection times from trajectory.
    
    Parameters
    ----------
    trajectory : ndarray
        AoI values over time
    
    Returns
    -------
    idt : ndarray
        Time between consecutive detections
    """
    det_times = detection_times(trajectory)
    if len(det_times) < 2:
        return np.array([], dtype=np.int64)
    return np.diff(det_times)


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
    return aoi_before - aoi_after
```

### Interface Summary

| Function | Input | Output |
|----------|-------|--------|
| `expected_aoi` | P_det | float |
| `expected_aoi_from_k` | k, R, L | float |
| `aoi_vectorized` | k_values, R, L | array |
| `simulate_aoi_trajectory` | P_det, T, seed | (T,) array |
| `simulate_aoi_trajectory_fast` | P_det, T, seed | (T,) array |
| `time_average_aoi` | trajectory | float |
| `peak_aoi` | trajectory | int |
| `detection_times` | trajectory | array |
| `inter_detection_times` | trajectory | array |
| `marginal_aoi_reduction` | k, R, L | float |

### Test Cases

```python
# test_aoi.py
import numpy as np
import pytest
from src.aoi import *

class TestExpectedAoI:
    def test_zero_detection(self):
        assert expected_aoi(0.0) == np.inf
    
    def test_certain_detection(self):
        assert expected_aoi(1.0) == 0.0
    
    def test_half_detection(self):
        assert expected_aoi(0.5) == 1.0
    
    def test_formula(self):
        P = 0.3
        expected = 1.0 / P - 1.0
        assert abs(expected_aoi(P) - expected) < 1e-10


class TestAoIFromK:
    def test_monotonicity(self):
        R, L = 10.0, 100.0
        aoi_values = [expected_aoi_from_k(k, R, L) for k in range(1, 101)]
        assert all(aoi_values[i] >= aoi_values[i+1] for i in range(99))
    
    def test_zero_volunteers(self):
        assert expected_aoi_from_k(0, 10.0, 100.0) == np.inf


class TestTrajectory:
    def test_shape(self):
        traj = simulate_aoi_trajectory(0.5, 1000, seed=42)
        assert traj.shape == (1000,)
    
    def test_non_negative(self):
        traj = simulate_aoi_trajectory(0.5, 1000, seed=42)
        assert np.all(traj >= 0)
    
    def test_convergence(self):
        """Test that simulated average converges to theoretical."""
        P_det = 0.3
        T = 100000
        traj = simulate_aoi_trajectory_fast(P_det, T, seed=42)
        simulated = time_average_aoi(traj)
        theoretical = expected_aoi(P_det)
        assert abs(simulated - theoretical) / theoretical < 0.05


class TestFastVsSlow:
    def test_equivalence(self):
        """Test that fast and slow implementations give same statistics."""
        P_det = 0.4
        T = 10000
        
        traj_slow = simulate_aoi_trajectory(P_det, T, seed=42)
        traj_fast = simulate_aoi_trajectory_fast(P_det, T, seed=42)
        
        # Same trajectory with same seed
        np.testing.assert_array_equal(traj_slow, traj_fast)
```

---

## 3.4 Game Theory Module (`game.py`)

### Purpose
Compute Nash equilibrium, social optimum, utilities, and Price of Anarchy.

### Specification

```python
"""
game.py - Game theory computations for crowd-finding
"""
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict
from .spatial import analytical_coverage_prob
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
    rho = np.pi * R**2 / L**2
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
    rho = np.pi * R**2 / L**2
    
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
            if u_active < u_deviate_inactive:
                continue
        
        # Condition 2: Inactive volunteers don't want to deviate
        if k < N:
            u_inactive = utility_inactive(k, R, L, B)
            u_deviate_active = utility_active(k + 1, R, L, B, c)
            if u_inactive < u_deviate_active:
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
    rho = np.pi * R**2 / L**2
    
    # Check boundary
    if c >= N * B * rho:
        return 0
    
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
        Price of Anarchy (>= 1)
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
    rho = np.pi * R**2 / L**2
    
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
```

### Interface Summary

| Function | Input | Output |
|----------|-------|--------|
| `utility_active` | k, R, L, B, c | float |
| `utility_inactive` | k, R, L, B | float |
| `marginal_utility` | k, R, L, B, c | float |
| `find_nash_equilibrium` | N, R, L, B, c | int |
| `social_welfare` | k, N, R, L, B, c | float |
| `find_social_optimum` | N, R, L, B, c | int |
| `price_of_anarchy` | N, R, L, B, c | float |
| `participation_gap` | N, R, L, B, c | int |
| `analyze_equilibrium` | N, R, L, B, c | dict |

### Test Cases

```python
# test_game.py
import numpy as np
import pytest
from src.game import *

class TestUtilities:
    def test_active_higher_cost(self):
        """Active utility < inactive utility when cost is high."""
        k, R, L, B, c = 10, 10.0, 100.0, 10.0, 100.0
        u_active = utility_active(k, R, L, B, c)
        u_inactive = utility_inactive(k - 1, R, L, B)
        assert u_active < u_inactive


class TestNashEquilibrium:
    def test_formula_matches_search(self):
        """Analytical NE matches exhaustive search."""
        for N in [50, 100, 200]:
            for c in [0.1, 1.0, 5.0]:
                k_formula = find_nash_equilibrium(N, 10.0, 100.0, 10.0, c)
                k_search = find_nash_equilibrium_search(N, 10.0, 100.0, 10.0, c)
                assert k_formula == k_search, f"Mismatch at N={N}, c={c}"
    
    def test_ne_is_equilibrium(self):
        """Verify NE satisfies equilibrium conditions."""
        N, R, L, B, c = 100, 10.0, 100.0, 10.0, 1.0
        k_star = find_nash_equilibrium(N, R, L, B, c)
        
        if k_star > 0:
            # Active don't want to deviate
            u_active = utility_active(k_star, R, L, B, c)
            u_deviate = utility_inactive(k_star - 1, R, L, B)
            assert u_active >= u_deviate - 1e-9
        
        if k_star < N:
            # Inactive don't want to deviate
            u_inactive = utility_inactive(k_star, R, L, B)
            u_deviate = utility_active(k_star + 1, R, L, B, c)
            assert u_inactive >= u_deviate - 1e-9


class TestSocialOptimum:
    def test_formula_matches_search(self):
        """Analytical optimum matches exhaustive search."""
        for N in [50, 100, 200]:
            for c in [0.1, 1.0, 5.0]:
                k_formula = find_social_optimum(N, 10.0, 100.0, 10.0, c)
                k_search = find_social_optimum_search(N, 10.0, 100.0, 10.0, c)
                assert k_formula == k_search, f"Mismatch at N={N}, c={c}"
    
    def test_optimum_geq_ne(self):
        """Social optimum >= Nash equilibrium."""
        for N in [50, 100, 200]:
            for c in [0.1, 1.0, 5.0]:
                k_star = find_nash_equilibrium(N, 10.0, 100.0, 10.0, c)
                k_opt = find_social_optimum(N, 10.0, 100.0, 10.0, c)
                assert k_opt >= k_star


class TestPoA:
    def test_poa_geq_one(self):
        """PoA >= 1 always."""
        for N in [50, 100]:
            for c in [0.5, 1.0, 2.0]:
                poa = price_of_anarchy(N, 10.0, 100.0, 10.0, c)
                assert poa >= 1.0 - 1e-9
```

---

## 3.5 Stackelberg Module (`stackelberg.py`)

### Purpose
Analyze Stackelberg game with platform incentives.

### Specification

```python
"""
stackelberg.py - Stackelberg game analysis for incentive design
"""
import numpy as np
from typing import Dict, Tuple, Optional
from .spatial import analytical_coverage_prob
from .game import find_nash_equilibrium, social_welfare, find_social_optimum


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
    c: float
) -> float:
    """
    Compute incentive needed to induce target participation k_target.
    
    p* = c - B * rho * (1 - rho)^(k_target - 1)
    
    Parameters
    ----------
    k_target : int
        Target number of active volunteers
    R, L, B, c : parameters
    
    Returns
    -------
    p : float
        Required incentive (may be negative if no incentive needed)
    """
    if k_target <= 0:
        return 0.0
    
    rho = np.pi * R**2 / L**2
    threshold = B * rho * (1 - rho) ** (k_target - 1)
    p_star = c - threshold
    
    return max(0, p_star)


def optimal_incentive(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float
) -> float:
    """
    Compute optimal incentive to implement social optimum.
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    
    Returns
    -------
    p_star : float
        Optimal incentive
    """
    k_opt = find_social_optimum(N, R, L, B, c)
    return optimal_incentive_for_target(k_opt, R, L, B, c)


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
    Compute platform's objective (welfare minus incentive cost).
    
    V(p) = W(k(p)) - p * k(p)
         = N * B * P_det(k) - k * c - p * k
         = N * B * P_det(k) - k * (c + p)
    
    Note: This equals social welfare since incentives are transfers.
    
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
    
    # Platform objective = social welfare (transfers cancel)
    return social_welfare(k, N, R, L, B, c)


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
    Find optimal incentive by grid search.
    
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


def analyze_stackelberg(
    N: int,
    R: float,
    L: float,
    B: float,
    c: float,
    budget: Optional[float] = None
) -> Dict:
    """
    Complete Stackelberg analysis.
    
    Parameters
    ----------
    N, R, L, B, c : parameters
    budget : float, optional
        Maximum total incentive
    
    Returns
    -------
    results : dict
        Comprehensive analysis results
    """
    # No-incentive baseline
    k_ne = find_nash_equilibrium(N, R, L, B, c)
    W_ne = social_welfare(k_ne, N, R, L, B, c)
    
    # Social optimum (unconstrained)
    k_opt = find_social_optimum(N, R, L, B, c)
    W_opt = social_welfare(k_opt, N, R, L, B, c)
    
    # Optimal incentive for social optimum
    p_opt = optimal_incentive(N, R, L, B, c)
    total_payment_opt = p_opt * k_opt
    
    # Check if budget allows social optimum
    budget_sufficient = budget is None or total_payment_opt <= budget
    
    if budget_sufficient:
        k_stack = k_opt
        p_stack = p_opt
        W_stack = W_opt
    else:
        # Find best within budget
        p_stack, k_stack, W_stack = find_optimal_incentive_search(
            N, R, L, B, c, budget=budget
        )
    
    return {
        'N': N,
        'R': R,
        'L': L,
        'B': B,
        'c': c,
        'budget': budget,
        # No incentive
        'k_ne': k_ne,
        'welfare_ne': W_ne,
        # Social optimum
        'k_opt': k_opt,
        'welfare_opt': W_opt,
        'p_for_optimum': p_opt,
        'cost_for_optimum': p_opt * k_opt,
        # Stackelberg solution
        'k_stackelberg': k_stack,
        'p_stackelberg': p_stack,
        'welfare_stackelberg': W_stack,
        'incentive_cost': p_stack * k_stack,
        # Metrics
        'budget_sufficient': budget_sufficient,
        'efficiency_gain': (W_stack - W_ne) / W_ne if W_ne > 0 else np.inf,
        'optimality_gap': (W_opt - W_stack) / W_opt if W_opt > 0 else 0,
    }


def incentive_sensitivity(
    N: int,
    R: float,
    L: float,
    B: float,
    c_values: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Analyze how optimal incentive varies with cost.
    
    Parameters
    ----------
    N, R, L, B : parameters
    c_values : array
        Cost values to analyze
    
    Returns
    -------
    results : dict of arrays
        Arrays of results for each cost value
    """
    n = len(c_values)
    results = {
        'c': c_values,
        'k_ne': np.zeros(n, dtype=int),
        'k_opt': np.zeros(n, dtype=int),
        'p_star': np.zeros(n),
        'total_incentive': np.zeros(n),
        'welfare_ne': np.zeros(n),
        'welfare_opt': np.zeros(n),
        'poa': np.zeros(n),
    }
    
    for i, c in enumerate(c_values):
        k_ne = find_nash_equilibrium(N, R, L, B, c)
        k_opt = find_social_optimum(N, R, L, B, c)
        p_star = optimal_incentive(N, R, L, B, c)
        
        results['k_ne'][i] = k_ne
        results['k_opt'][i] = k_opt
        results['p_star'][i] = p_star
        results['total_incentive'][i] = p_star * k_opt
        results['welfare_ne'][i] = social_welfare(k_ne, N, R, L, B, c)
        results['welfare_opt'][i] = social_welfare(k_opt, N, R, L, B, c)
        
        if results['welfare_ne'][i] > 0:
            results['poa'][i] = results['welfare_opt'][i] / results['welfare_ne'][i]
        else:
            results['poa'][i] = np.inf
    
    return results
```

### Interface Summary

| Function | Input | Output |
|----------|-------|--------|
| `induced_equilibrium` | p, N, R, L, B, c | int |
| `optimal_incentive_for_target` | k_target, R, L, B, c | float |
| `optimal_incentive` | N, R, L, B, c | float |
| `total_incentive_cost` | p, N, R, L, B, c | float |
| `platform_objective` | p, N, R, L, B, c, budget | float |
| `find_optimal_incentive_search` | N, R, L, B, c, budget | (p, k, welfare) |
| `analyze_stackelberg` | N, R, L, B, c, budget | dict |
| `incentive_sensitivity` | N, R, L, B, c_values | dict of arrays |

---

## 3.6 Simulation Module (`simulation.py`)

### Purpose
Monte Carlo simulation engine for validation and extended analysis.

### Specification

```python
"""
simulation.py - Monte Carlo simulation engine
"""
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from .config import SimConfig
from .spatial import generate_positions, generate_target, compute_distances, coverage_mask
from .aoi import time_average_aoi, peak_aoi


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
    """
    
    def __init__(self, config: SimConfig):
        """
        Initialize simulation.
        
        Parameters
        ----------
        config : SimConfig
            Simulation configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.simulation.seed)
        self.reset()
    
    def reset(self, seed: Optional[int] = None):
        """Reset simulation state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        self.aoi = 0
        self.trajectory = []
        self.detection_times = []
    
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
        # Generate volunteer positions (those who are active)
        if k > 0:
            positions = self.rng.uniform(
                0, self.config.L, size=(k, 2)
            )
            target = self.rng.uniform(0, self.config.L, size=2)
            
            # Check for detection
            distances = np.linalg.norm(positions - target, axis=1)
            detected = np.any(distances <= self.config.R)
        else:
            detected = False
        
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
        
        for _ in range(self.config.simulation.T):
            self.step(k)
        
        trajectory = np.array(self.trajectory)
        
        return SimulationResult(
            mean_aoi=float(np.mean(trajectory)),
            peak_aoi=int(np.max(trajectory)),
            detection_rate=len(self.detection_times) / self.config.simulation.T,
            n_detections=len(self.detection_times),
            trajectory=trajectory if store_trajectory else None,
        )


class MonteCarloSimulation:
    """
    Monte Carlo simulation runner.
    """
    
    def __init__(self, config: SimConfig):
        """
        Initialize Monte Carlo runner.
        
        Parameters
        ----------
        config : SimConfig
            Simulation configuration
        """
        self.config = config
        self.base_seed = config.simulation.seed or 0
    
    def run(
        self,
        k: int,
        n_runs: Optional[int] = None,
        show_progress: bool = True,
        store_individual: bool = False
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Parameters
        ----------
        k : int
            Number of active volunteers
        n_runs : int, optional
            Number of runs (uses config default if not specified)
        show_progress : bool
            Show progress bar
        store_individual : bool
            Store individual run results
        
        Returns
        -------
        result : MonteCarloResult
        """
        if n_runs is None:
            n_runs = self.config.simulation.n_runs
        
        mean_aois = []
        peak_aois = []
        detection_rates = []
        individual_results = [] if store_individual else None
        
        iterator = range(n_runs)
        if show_progress:
            iterator = tqdm(iterator, desc=f"MC simulation (k={k})")
        
        for i in iterator:
            # Create simulation with unique seed
            run_config = SimConfig(
                physical=self.config.physical,
                game=self.config.game,
                simulation=type(self.config.simulation)(
                    T=self.config.simulation.T,
                    n_runs=1,
                    seed=self.base_seed + i,
                )
            )
            
            sim = Simulation(run_config)
            result = sim.run(k, store_trajectory=store_individual)
            
            mean_aois.append(result.mean_aoi)
            peak_aois.append(result.peak_aoi)
            detection_rates.append(result.detection_rate)
            
            if store_individual:
                individual_results.append(result)
        
        return MonteCarloResult(
            mean_aoi=float(np.mean(mean_aois)),
            mean_aoi_std=float(np.std(mean_aois)),
            mean_peak_aoi=float(np.mean(peak_aois)),
            mean_detection_rate=float(np.mean(detection_rates)),
            n_runs=n_runs,
            individual_results=individual_results,
        )


def run_parameter_sweep(
    base_config: SimConfig,
    k_values: List[int],
    n_runs: int = 100,
    show_progress: bool = True
) -> Dict[str, NDArray]:
    """
    Run simulations across range of k values.
    
    Parameters
    ----------
    base_config : SimConfig
        Base configuration
    k_values : list of int
        Values of k to simulate
    n_runs : int
        Runs per k value
    show_progress : bool
        Show progress
    
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
    
    mc = MonteCarloSimulation(base_config)
    
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
    config: SimConfig,
    k_values: List[int],
    n_runs: int = 1000,
    tolerance: float = 0.1
) -> Dict:
    """
    Validate analytical formulas against simulation.
    
    Parameters
    ----------
    config : SimConfig
        Configuration
    k_values : list of int
        Values of k to test
    n_runs : int
        Runs per k value
    tolerance : float
        Relative tolerance for validation
    
    Returns
    -------
    validation : dict
        Validation results
    """
    from .aoi import expected_aoi_from_k
    from .spatial import analytical_coverage_prob
    
    results = run_parameter_sweep(config, k_values, n_runs, show_progress=True)
    
    validation = {
        'k': np.array(k_values),
        'analytical_aoi': np.zeros(len(k_values)),
        'simulated_aoi': results['mean_aoi'],
        'simulated_aoi_std': results['mean_aoi_std'],
        'relative_error': np.zeros(len(k_values)),
        'passed': np.zeros(len(k_values), dtype=bool),
    }
    
    for i, k in enumerate(k_values):
        analytical = expected_aoi_from_k(k, config.R, config.L)
        validation['analytical_aoi'][i] = analytical
        
        if analytical > 0:
            error = abs(results['mean_aoi'][i] - analytical) / analytical
        else:
            error = abs(results['mean_aoi'][i])
        
        validation['relative_error'][i] = error
        validation['passed'][i] = error < tolerance
    
    validation['all_passed'] = np.all(validation['passed'])
    
    return validation
```

### Interface Summary

| Class/Function | Purpose |
|----------------|---------|
| `Simulation` | Single simulation run |
| `MonteCarloSimulation` | Multiple run aggregation |
| `run_parameter_sweep` | Sweep over k values |
| `validate_analytical` | Compare simulation to theory |

---

## 3.7 Visualization Module (`visualization.py`)

### Specification

```python
"""
visualization.py - Publication-quality plotting utilities
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Tuple
import os


# IEEE single-column style
STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.5, 2.5),
    'figure.dpi': 150,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def apply_style():
    """Apply IEEE-style formatting."""
    plt.rcParams.update(STYLE)


def save_figure(fig: Figure, name: str, output_dir: str = 'results/figures'):
    """Save figure in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
    
    print(f"Saved: {name}")


def plot_equilibrium_vs_cost(
    c_values: np.ndarray,
    k_ne: np.ndarray,
    k_opt: np.ndarray,
    params: Dict,
    save: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plot equilibrium participation vs cost.
    
    Parameters
    ----------
    c_values : array
        Cost values
    k_ne : array
        Nash equilibrium participation
    k_opt : array
        Social optimum participation
    params : dict
        Parameter info for title
    save : bool
        Whether to save figure
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots()
    
    ax.plot(c_values, k_ne, 'b-o', label='Nash Equilibrium $k^*$')
    ax.plot(c_values, k_opt, 'r--s', label='Social Optimum $k^{opt}$')
    
    ax.set_xlabel('Cost $c$')
    ax.set_ylabel('Active Volunteers $k$')
    ax.legend()
    ax.set_title(f"N={params['N']}, R/L={params['R']/params['L']:.2f}")
    
    if save:
        save_figure(fig, 'equilibrium_vs_cost')
    
    return fig, ax


def plot_aoi_vs_k(
    k_values: np.ndarray,
    aoi_analytical: np.ndarray,
    aoi_simulated: Optional[np.ndarray] = None,
    aoi_std: Optional[np.ndarray] = None,
    params: Dict = None,
    save: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plot AoI vs number of active volunteers.
    
    Parameters
    ----------
    k_values : array
        Number of active volunteers
    aoi_analytical : array
        Analytical AoI values
    aoi_simulated : array, optional
        Simulated AoI values
    aoi_std : array, optional
        Standard deviation of simulated AoI
    params : dict
        Parameter info
    save : bool
        Whether to save
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots()
    
    ax.plot(k_values, aoi_analytical, 'b-', label='Analytical', linewidth=2)
    
    if aoi_simulated is not None:
        if aoi_std is not None:
            ax.errorbar(k_values, aoi_simulated, yerr=aoi_std,
                       fmt='ro', label='Simulated', capsize=3, markersize=4)
        else:
            ax.plot(k_values, aoi_simulated, 'ro', label='Simulated')
    
    ax.set_xlabel('Active Volunteers $k$')
    ax.set_ylabel('Expected AoI $\\bar{\\Delta}$')
    ax.legend()
    ax.set_yscale('log')
    
    if save:
        save_figure(fig, 'aoi_vs_k')
    
    return fig, ax


def plot_poa_heatmap(
    N_values: np.ndarray,
    c_values: np.ndarray,
    poa_matrix: np.ndarray,
    params: Dict = None,
    save: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plot Price of Anarchy heatmap.
    
    Parameters
    ----------
    N_values : array
        Population sizes
    c_values : array
        Cost values
    poa_matrix : 2D array
        PoA values (rows=N, cols=c)
    params : dict
        Parameter info
    save : bool
        Whether to save
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    im = ax.imshow(poa_matrix, aspect='auto', origin='lower',
                   extent=[c_values[0], c_values[-1], N_values[0], N_values[-1]],
                   cmap='YlOrRd')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Price of Anarchy')
    
    ax.set_xlabel('Cost $c$')
    ax.set_ylabel('Population $N$')
    ax.set_title('Price of Anarchy')
    
    if save:
        save_figure(fig, 'poa_heatmap')
    
    return fig, ax


def plot_stackelberg_incentive(
    c_values: np.ndarray,
    p_star: np.ndarray,
    total_incentive: np.ndarray,
    params: Dict = None,
    save: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plot optimal Stackelberg incentive vs cost.
    
    Parameters
    ----------
    c_values : array
        Cost values
    p_star : array
        Optimal per-volunteer incentive
    total_incentive : array
        Total platform payment
    params : dict
        Parameter info
    save : bool
        Whether to save
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    
    # Per-volunteer incentive
    ax1.plot(c_values, p_star, 'b-o')
    ax1.plot(c_values, c_values, 'k--', alpha=0.5, label='$p = c$')
    ax1.set_xlabel('Cost $c$')
    ax1.set_ylabel('Optimal Incentive $p^*$')
    ax1.legend()
    ax1.set_title('Per-Volunteer Incentive')
    
    # Total incentive
    ax2.plot(c_values, total_incentive, 'r-s')
    ax2.set_xlabel('Cost $c$')
    ax2.set_ylabel('Total Payment $p^* \\cdot k^{opt}$')
    ax2.set_title('Platform Total Cost')
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'stackelberg_incentive')
    
    return fig, (ax1, ax2)


def plot_welfare_comparison(
    c_values: np.ndarray,
    W_ne: np.ndarray,
    W_opt: np.ndarray,
    W_stack: np.ndarray,
    params: Dict = None,
    save: bool = True
) -> Tuple[Figure, Axes]:
    """
    Compare welfare under different mechanisms.
    
    Parameters
    ----------
    c_values : array
        Cost values
    W_ne : array
        Nash equilibrium welfare
    W_opt : array
        Social optimum welfare
    W_stack : array
        Stackelberg welfare
    params : dict
        Parameter info
    save : bool
        Whether to save
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots()
    
    ax.plot(c_values, W_ne, 'b-o', label='Nash Equilibrium')
    ax.plot(c_values, W_opt, 'g--', label='Social Optimum', linewidth=2)
    ax.plot(c_values, W_stack, 'r-s', label='Stackelberg')
    
    ax.set_xlabel('Cost $c$')
    ax.set_ylabel('Social Welfare $W$')
    ax.legend()
    ax.set_title('Welfare Comparison')
    
    if save:
        save_figure(fig, 'welfare_comparison')
    
    return fig, ax


def plot_spatial_snapshot(
    positions: np.ndarray,
    target: np.ndarray,
    R: float,
    L: float,
    active_mask: Optional[np.ndarray] = None,
    save: bool = True
) -> Tuple[Figure, Axes]:
    """
    Visualize spatial configuration.
    
    Parameters
    ----------
    positions : array (N, 2)
        Volunteer positions
    target : array (2,)
        Target position
    R : float
        Detection radius
    L : float
        Area side
    active_mask : bool array, optional
        Which volunteers are active
    save : bool
        Whether to save
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Target and detection zone
    circle = plt.Circle(target, R, fill=False, color='red', 
                        linestyle='--', linewidth=2, label='Detection Zone')
    ax.add_patch(circle)
    ax.plot(target[0], target[1], 'r*', markersize=15, label='Target')
    
    # Volunteers
    if active_mask is None:
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c='blue', s=20, alpha=0.6, label='Volunteers')
    else:
        ax.scatter(positions[~active_mask, 0], positions[~active_mask, 1],
                  c='gray', s=15, alpha=0.4, label='Inactive')
        ax.scatter(positions[active_mask, 0], positions[active_mask, 1],
                  c='blue', s=25, alpha=0.8, label='Active')
    
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_title('Spatial Configuration')
    
    if save:
        save_figure(fig, 'spatial_snapshot')
    
    return fig, ax
```

---

# 4. Implementation Pipeline

## 4.1 Day-by-Day Schedule

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| **1** | Setup environment, `config.py`, `spatial.py` | Working geometry tests pass |
| **2** | `aoi.py`, tests | AoI computation validated |
| **3** | `game.py`, tests | NE solver verified |
| **4** | `simulation.py` | Monte Carlo engine working |
| **5** | `stackelberg.py`, `visualization.py` | Complete module set |
| **6** | Experiments 1-3 | Core results |
| **7** | Experiment 4-5, polish | All figures ready |

## 4.2 Implementation Order

```
1. config.py          [Foundation]
       ↓
2. spatial.py         [Geometry]
       ↓
3. aoi.py             [Core metric]
       ↓
4. game.py            [Theory]
       ↓
5. simulation.py      [Validation]
       ↓
6. stackelberg.py     [Extension]
       ↓
7. visualization.py   [Output]
       ↓
8. experiments/       [Results]
```

## 4.3 Git Workflow

```bash
# Initial setup
git init
git add .
git commit -m "Initial project structure"

# Feature branches
git checkout -b feature/spatial
# ... implement ...
git add src/spatial.py tests/test_spatial.py
git commit -m "Add spatial module with coverage functions"
git checkout main
git merge feature/spatial

# Continue for each module
```

---

# 5. Testing Strategy

## 5.1 Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit | Individual functions | `tests/test_*.py` |
| Integration | Module interactions | `tests/test_integration.py` |
| Validation | Theory vs simulation | `experiments/validate.py` |

## 5.2 Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_game.py -v

# Run specific test
pytest tests/test_game.py::TestNashEquilibrium::test_formula_matches_search -v
```

## 5.3 Coverage Target

- Minimum: 80%
- Target: 90%+
- Critical modules (game.py, aoi.py): 95%+

---

# 6. Experiment Design

## 6.1 Experiment 1: Equilibrium Analysis

**Objective:** Characterize NE and social optimum as functions of parameters.

**Parameters:**
```python
N_values = [50, 100, 200, 500]
c_values = np.linspace(0.1, 10.0, 50)
R_L_ratios = [0.05, 0.10, 0.15, 0.20]
B = 10.0
L = 100.0
```

**Outputs:**
- `k_star` vs `c` curves for different N
- `k_opt` vs `c` curves
- Participation gap analysis

## 6.2 Experiment 2: Price of Anarchy

**Objective:** Quantify inefficiency of selfish behavior.

**Analysis:**
- PoA vs c for fixed N
- PoA vs N for fixed c
- PoA heatmap over (N, c) grid

## 6.3 Experiment 3: Stackelberg Incentives

**Objective:** Design optimal platform incentive.

**Analysis:**
- Optimal p* vs c
- Total incentive cost vs c
- Welfare improvement from incentives

## 6.4 Experiment 4: Simulation Validation

**Objective:** Verify analytical results.

**Method:**
- Compare analytical AoI to simulated
- Statistical tests for agreement
- Confidence intervals

## 6.5 Experiment 5: Sensitivity Analysis

**Objective:** Understand parameter importance.

**Parameters varied:**
- R/L ratio (coverage density)
- B/c ratio (benefit-to-cost)
- N (population size)

---

# 7. Results Analysis

## 7.1 Expected Findings

1. **Under-participation:** $k^* < k^{\text{opt}}$ always
2. **Gap scaling:** $\Delta k \sim \ln(N) / \rho$
3. **PoA bounded:** PoA converges to 1 for large N
4. **Incentive cost:** $p^* \approx c(1 - 1/N)$ for large N

## 7.2 Validation Criteria

| Metric | Criterion |
|--------|-----------|
| AoI relative error | < 5% |
| NE formula vs search | Exact match |
| Social optimum formula vs search | Exact match |
| PoA | ≥ 1 always |

## 7.3 Output Files

```
results/
├── data/
│   ├── equilibrium_analysis.csv
│   ├── poa_grid.csv
│   ├── stackelberg_analysis.csv
│   └── validation_results.csv
└── figures/
    ├── equilibrium_vs_cost.pdf
    ├── aoi_vs_k.pdf
    ├── poa_heatmap.pdf
    ├── stackelberg_incentive.pdf
    ├── welfare_comparison.pdf
    └── spatial_snapshot.pdf
```

---

# 8. Visualization Guidelines

## 8.1 Figure Specifications

| Property | Value |
|----------|-------|
| Width (single column) | 3.5 inches |
| Width (double column) | 7.0 inches |
| Font | Serif (Times-like) |
| Font size | 10pt (labels), 8pt (ticks) |
| Line width | 1.5pt |
| Marker size | 4pt |
| Format | PDF (vector) + PNG (preview) |

## 8.2 Color Scheme

```python
COLORS = {
    'nash': '#1f77b4',      # Blue
    'optimum': '#2ca02c',   # Green
    'stackelberg': '#d62728', # Red
    'simulated': '#ff7f0e', # Orange
    'inactive': '#7f7f7f',  # Gray
}
```

## 8.3 Required Figures

1. **equilibrium_vs_cost.pdf**: k* and k^opt vs c
2. **aoi_vs_k.pdf**: AoI vs k with analytical and simulated
3. **poa_heatmap.pdf**: PoA over parameter space
4. **stackelberg_incentive.pdf**: p* and total cost vs c
5. **welfare_comparison.pdf**: W under different mechanisms
6. **spatial_snapshot.pdf**: Example spatial configuration

---

# 9. Execution Checklist

## 9.1 Pre-Implementation

- [ ] Create directory structure
- [ ] Setup virtual environment
- [ ] Install dependencies
- [ ] Initialize git repository

## 9.2 Implementation

- [ ] `config.py` complete and tested
- [ ] `spatial.py` complete and tested
- [ ] `aoi.py` complete and tested
- [ ] `game.py` complete and tested
- [ ] `simulation.py` complete and tested
- [ ] `stackelberg.py` complete and tested
- [ ] `visualization.py` complete

## 9.3 Validation

- [ ] All unit tests pass (>90% coverage)
- [ ] Analytical formulas match search algorithms
- [ ] Simulation matches analytical predictions (<5% error)
- [ ] PoA ≥ 1 verified across all parameters

## 9.4 Experiments

- [ ] Experiment 1: Equilibrium analysis complete
- [ ] Experiment 2: PoA analysis complete
- [ ] Experiment 3: Stackelberg analysis complete
- [ ] Experiment 4: Validation complete
- [ ] Experiment 5: Sensitivity analysis complete

## 9.5 Output

- [ ] All figures generated
- [ ] All data files saved
- [ ] Results documented
- [ ] Code cleaned and commented

---

# Appendix: Quick Reference

## Key Formulas

| Formula | Expression |
|---------|------------|
| Coverage ratio | $\rho = \pi R^2 / L^2$ |
| Detection probability | $P_{\text{det}}(k) = 1 - (1-\rho)^k$ |
| Expected AoI | $\bar{\Delta}(k) = (1-\rho)^k / [1-(1-\rho)^k]$ |
| Utility (active) | $U = B \cdot P_{\text{det}}(k) - c$ |
| Nash equilibrium | $k^* = \lfloor 1 + \ln(c/B\rho) / \ln(1-\rho) \rfloor$ |
| Social optimum | $k^{\text{opt}} = \lfloor 1 + \ln(c/NB\rho) / \ln(1-\rho) \rfloor$ |
| Optimal incentive | $p^* = c - B\rho(1-\rho)^{k^{\text{opt}}-1}$ |

## Command Reference

```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific experiment
python experiments/exp01_equilibrium_analysis.py

# Run all experiments
python experiments/run_all.py

# Generate figures only
python -c "from experiments.run_all import generate_figures; generate_figures()"
```

---

*Document Version: 1.0*
*Last Updated: December 2024*
