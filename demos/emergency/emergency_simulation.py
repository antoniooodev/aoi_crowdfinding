"""
emergency_simulation.py - Final Corrected Simulation

Version 2.0: Added heterogeneous cost model support.

TWO MODES:
1. STATIC: i.i.d. positions each step → validates P_det formula exactly
2. DYNAMIC: agent movement → shows qualitative under-participation effect

TWO COST MODELS:
1. HOMOGENEOUS: All volunteers have same cost c (original)
2. HETEROGENEOUS: Volunteers have costs c_i ~ F[c_min, c_max] (new)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HeterogeneousCostParams
from src.game import (
    find_nash_heterogeneous,
    find_social_optimum_heterogeneous_expected,
    analyze_equilibrium_heterogeneous,
    price_of_anarchy_heterogeneous,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Simulation configuration."""
    L: float = 500.0          # Area side
    N: int = 100              # Volunteers
    R: float = 30.0           # Detection radius
    B: float = 10.0           # Benefit
    T_max: int = 600          # Max steps
    
    # Dynamic mode parameters
    volunteer_speed: float = 1.0
    target_speed: float = 0.3
    
    # Rescue parameters
    response_time: int = 60     # Steps needed to complete rescue
    aoi_threshold: int = 30     # Max AoI during rescue
    
    # Cost model: homogeneous (single c) or heterogeneous (c_min, c_max)
    use_heterogeneous: bool = False
    c_min: Optional[float] = None
    c_max: Optional[float] = None
    cost_distribution: str = "uniform"
    
    @property
    def rho(self) -> float:
        return np.pi * self.R**2 / self.L**2
    
    @property
    def B_rho(self) -> float:
        return self.B * self.rho
    
    @property
    def NB_rho(self) -> float:
        return self.N * self.B * self.rho
    
    def get_cost_params(self, c: float) -> HeterogeneousCostParams:
        """Get cost parameters for heterogeneous model."""
        if self.use_heterogeneous and self.c_min is not None and self.c_max is not None:
            return HeterogeneousCostParams(
                c_min=self.c_min,
                c_max=self.c_max,
                distribution=self.cost_distribution
            )
        else:
            # Use c as mean, create spread around it
            spread_ratio = 2.0  # c_max/c_min = 2
            c_min = 2 * c / (1 + spread_ratio)
            c_max = spread_ratio * c_min
            return HeterogeneousCostParams(c_min=c_min, c_max=c_max)


# =============================================================================
# HOMOGENEOUS MODEL FUNCTIONS (original)
# =============================================================================

def compute_k_star(N: int, rho: float, B: float, c: float) -> int:
    """Nash equilibrium (homogeneous cost)."""
    if c >= B * rho:
        return 0
    if c <= 0:
        return N
    k = 1 + np.log(c / (B * rho)) / np.log(1 - rho)
    return max(0, min(N, int(np.floor(k))))


def compute_k_opt(N: int, rho: float, B: float, c: float) -> int:
    """Social optimum (homogeneous cost)."""
    if c >= N * B * rho:
        return 0
    if c <= 0:
        return N
    k = 1 + np.log(c / (N * B * rho)) / np.log(1 - rho)
    return max(0, min(N, int(np.floor(k))))


def analytical_P_det(k: int, rho: float) -> float:
    """Theoretical detection probability."""
    if k <= 0:
        return 0.0
    return 1.0 - (1.0 - rho)**k


def analytical_aoi(k: int, rho: float) -> float:
    """Theoretical expected AoI."""
    P = analytical_P_det(k, rho)
    if P <= 0:
        return np.inf
    return 1.0 / P - 1.0


# =============================================================================
# HETEROGENEOUS MODEL FUNCTIONS (new)
# =============================================================================

def compute_k_star_heterogeneous(N: int, rho: float, B: float, 
                                  cost_params: HeterogeneousCostParams) -> int:
    """Nash equilibrium (heterogeneous costs)."""
    k_star, _ = find_nash_heterogeneous(N, rho, B, cost_params)
    return k_star


def compute_k_opt_heterogeneous(N: int, rho: float, B: float,
                                 cost_params: HeterogeneousCostParams) -> int:
    """Social optimum (heterogeneous costs)."""
    k_opt, _ = find_social_optimum_heterogeneous_expected(N, rho, B, cost_params)
    return k_opt


# =============================================================================
# STATIC SIMULATION (for P_det validation)
# =============================================================================

def simulate_static(config: Config, k: int, n_steps: int, seed: int = 42) -> Dict:
    """
    Static simulation: i.i.d. positions each step.
    This EXACTLY matches the theoretical model.
    """
    rng = np.random.default_rng(seed)
    
    detections = 0
    aoi = 0
    aoi_history = []
    
    # Target in interior
    margin = config.R
    target_x = rng.uniform(margin, config.L - margin)
    target_y = rng.uniform(margin, config.L - margin)
    
    for _ in range(n_steps):
        # Fresh i.i.d. positions each step (matches theory exactly)
        if k > 0:
            vol_x = rng.uniform(0, config.L, size=k)
            vol_y = rng.uniform(0, config.L, size=k)
            distances = np.sqrt((vol_x - target_x)**2 + (vol_y - target_y)**2)
            detected = np.any(distances <= config.R)
        else:
            detected = False
        
        if detected:
            aoi = 0
            detections += 1
        else:
            aoi += 1
        
        aoi_history.append(aoi)
    
    return {
        'detections': detections,
        'detection_rate': detections / n_steps,
        'mean_aoi': np.mean(aoi_history),
        'max_aoi': np.max(aoi_history),
    }


def validate_P_det_static(config: Config, k_values: List[int], 
                          n_runs: int = 100, n_steps: int = 1000) -> Dict:
    """
    Validate P_det using STATIC simulation (i.i.d. positions).
    This should match theory within Monte Carlo error.
    """
    print("="*60)
    print("VALIDATION: P_det with STATIC simulation (i.i.d. positions)")
    print("="*60)
    print(f"ρ = {config.rho:.6f}, Interior target sampling")
    print()
    
    results = {'k': [], 'theory': [], 'sim_mean': [], 'sim_std': [], 'error': []}
    
    for k in k_values:
        theory = analytical_P_det(k, config.rho)
        
        rates = []
        for run in range(n_runs):
            res = simulate_static(config, k, n_steps, seed=run)
            rates.append(res['detection_rate'])
        
        sim_mean = np.mean(rates)
        sim_std = np.std(rates) / np.sqrt(n_runs)
        error = abs(sim_mean - theory) / theory if theory > 0 else 0
        
        results['k'].append(k)
        results['theory'].append(theory)
        results['sim_mean'].append(sim_mean)
        results['sim_std'].append(sim_std)
        results['error'].append(error)
        
        status = "✓" if error < 0.05 else "~" if error < 0.10 else "✗"
        print(f"k={k:3d}: theory={theory:.4f}, sim={sim_mean:.4f}±{sim_std:.4f}, "
              f"error={error*100:.1f}% {status}")
    
    passed = sum(1 for e in results['error'] if e < 0.10)
    print(f"\nPassed: {passed}/{len(k_values)} (<10% error)")
    
    return results


# =============================================================================
# DYNAMIC SIMULATION (for qualitative demonstration)
# =============================================================================

class DynamicSimulation:
    """
    Dynamic agent-based simulation.
    
    NOTE: This does NOT validate the exact formula because:
    - Positions are correlated across time (agents move)
    - This is intentional: we want to show the QUALITATIVE effect
    """
    
    def __init__(self, config: Config, k: int, seed: int = 42):
        self.config = config
        self.k = k
        self.rng = np.random.default_rng(seed)
        
        # Volunteers
        self.vol_x = self.rng.uniform(0, config.L, config.N)
        self.vol_y = self.rng.uniform(0, config.L, config.N)
        self.vol_active = np.zeros(config.N, dtype=bool)
        self.vol_active[:k] = True
        self.rng.shuffle(self.vol_active)
        
        self.waypoints_x = self.rng.uniform(0, config.L, config.N)
        self.waypoints_y = self.rng.uniform(0, config.L, config.N)
        
        # Target (interior)
        margin = config.R
        self.target_x = self.rng.uniform(margin, config.L - margin)
        self.target_y = self.rng.uniform(margin, config.L - margin)
        self.target_vx = 0.0
        self.target_vy = 0.0
        
        # State
        self.time = 0
        self.aoi = 0
        self.aoi_history = []
        self.detections = 0
        
        # Rescue state
        self.rescue_started = False
        self.rescue_start_time = None
        self.success = False
    
    def step(self):
        """One simulation step."""
        # Move volunteers
        for i in range(self.config.N):
            if not self.vol_active[i]:
                continue
            
            dx = self.waypoints_x[i] - self.vol_x[i]
            dy = self.waypoints_y[i] - self.vol_y[i]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 5:
                self.waypoints_x[i] = self.rng.uniform(0, self.config.L)
                self.waypoints_y[i] = self.rng.uniform(0, self.config.L)
            else:
                step = self.config.volunteer_speed
                self.vol_x[i] += (dx/dist) * min(step, dist)
                self.vol_y[i] += (dy/dist) * min(step, dist)
        
        # Move target (random walk)
        if self.rng.random() < 0.1:  # Change direction occasionally
            angle = self.rng.uniform(0, 2*np.pi)
            self.target_vx = self.config.target_speed * np.cos(angle)
            self.target_vy = self.config.target_speed * np.sin(angle)
        
        new_x = self.target_x + self.target_vx
        new_y = self.target_y + self.target_vy
        
        # Bounce off walls
        margin = self.config.R
        if new_x < margin or new_x > self.config.L - margin:
            self.target_vx *= -1
            new_x = np.clip(new_x, margin, self.config.L - margin)
        if new_y < margin or new_y > self.config.L - margin:
            self.target_vy *= -1
            new_y = np.clip(new_y, margin, self.config.L - margin)
        
        self.target_x = new_x
        self.target_y = new_y
        
        # Check detection
        detected = False
        for i in range(self.config.N):
            if not self.vol_active[i]:
                continue
            dist = np.sqrt((self.vol_x[i] - self.target_x)**2 + 
                          (self.vol_y[i] - self.target_y)**2)
            if dist <= self.config.R:
                detected = True
                break
        
        # Update AoI
        if detected:
            self.aoi = 0
            self.detections += 1
        else:
            self.aoi += 1
        
        self.aoi_history.append(self.aoi)
        
        # Rescue logic
        if not self.rescue_started and detected:
            self.rescue_started = True
            self.rescue_start_time = self.time
        
        if self.rescue_started:
            elapsed = self.time - self.rescue_start_time
            if elapsed >= self.config.response_time:
                if self.aoi <= self.config.aoi_threshold:
                    self.success = True
        
        self.time += 1
    
    def run(self) -> Dict:
        """Run until success or timeout."""
        while self.time < self.config.T_max and not self.success:
            self.step()
        
        return {
            'success': self.success,
            'time': self.time,
            'detections': self.detections,
            'mean_aoi': np.mean(self.aoi_history) if self.aoi_history else np.inf,
            'max_aoi': np.max(self.aoi_history) if self.aoi_history else np.inf,
        }


# =============================================================================
# INFORMATIVE COST SWEEP - HOMOGENEOUS (original)
# =============================================================================

def run_informative_sweep_homogeneous(config: Config, n_runs: int = 50) -> Dict:
    """
    Sweep cost in region where BOTH k* and k_opt vary (homogeneous cost).
    """
    print("\n" + "="*60)
    print("HOMOGENEOUS COST SWEEP")
    print("="*60)
    
    rho = config.rho
    B_rho = config.B_rho
    NB_rho = config.NB_rho
    
    print(f"Bρ = {B_rho:.4f} (threshold for k* > 0)")
    print(f"NBρ = {NB_rho:.4f} (threshold for k_opt > 0)")
    
    c_values = np.linspace(0.1 * B_rho, 0.95 * B_rho, 12)
    
    results = {
        'c': [], 'c_norm': [],
        'k_star': [], 'k_opt': [], 'gap': [],
        'P_det_star': [], 'P_det_opt': [],
        'aoi_star': [], 'aoi_opt': [],
        'success_star': [], 'success_opt': [],
    }
    
    print(f"\n{'c/(Bρ)':<8} {'k*':<5} {'k_opt':<6} {'Gap':<5} "
          f"{'P*(th)':<8} {'AoI*(th)':<10} {'Succ*%':<8} {'Succ_opt%':<8}")
    print("-"*75)
    
    for c in c_values:
        k_star = compute_k_star(config.N, rho, config.B, c)
        k_opt = compute_k_opt(config.N, rho, config.B, c)
        
        P_star = analytical_P_det(k_star, rho)
        P_opt = analytical_P_det(k_opt, rho)
        aoi_star = analytical_aoi(k_star, rho)
        aoi_opt = analytical_aoi(k_opt, rho)
        
        # Run dynamic simulations
        success_star = 0
        success_opt = 0
        
        for run in range(n_runs):
            sim = DynamicSimulation(config, k_star, seed=run)
            if sim.run()['success']:
                success_star += 1
            
            sim = DynamicSimulation(config, k_opt, seed=run + 10000)
            if sim.run()['success']:
                success_opt += 1
        
        results['c'].append(c)
        results['c_norm'].append(c / B_rho)
        results['k_star'].append(k_star)
        results['k_opt'].append(k_opt)
        results['gap'].append(k_opt - k_star)
        results['P_det_star'].append(P_star)
        results['P_det_opt'].append(P_opt)
        results['aoi_star'].append(aoi_star)
        results['aoi_opt'].append(aoi_opt)
        results['success_star'].append(success_star / n_runs)
        results['success_opt'].append(success_opt / n_runs)
        
        print(f"{c/B_rho:<8.2f} {k_star:<5} {k_opt:<6} {k_opt-k_star:<5} "
              f"{P_star:<8.3f} {aoi_star:<10.1f} "
              f"{100*success_star/n_runs:<8.1f} {100*success_opt/n_runs:<8.1f}")
    
    return results


# =============================================================================
# INFORMATIVE COST SWEEP - HETEROGENEOUS (new)
# =============================================================================

def run_informative_sweep_heterogeneous(config: Config, n_runs: int = 50,
                                         spread_ratio: float = 2.0) -> Dict:
    """
    Sweep cost in region where BOTH k* and k_opt vary (heterogeneous costs).
    
    Parameters
    ----------
    config : Config
        Simulation configuration
    n_runs : int
        Number of Monte Carlo runs per configuration
    spread_ratio : float
        Heterogeneity ratio c_max/c_min (fixed across sweep)
    """
    print("\n" + "="*60)
    print(f"HETEROGENEOUS COST SWEEP (c_max/c_min = {spread_ratio})")
    print("="*60)
    
    rho = config.rho
    B_rho = config.B_rho
    NB_rho = config.NB_rho
    
    print(f"Bρ = {B_rho:.4f} (threshold for k* > 0)")
    print(f"NBρ = {NB_rho:.4f} (threshold for k_opt > 0)")
    print(f"Heterogeneity ratio: {spread_ratio}")
    
    # Sweep mean cost
    mean_costs = np.linspace(0.1 * B_rho, 0.95 * B_rho, 12)
    
    results = {
        'mean_cost': [], 'c_norm': [],
        'c_min': [], 'c_max': [],
        'k_star': [], 'k_opt': [], 'gap': [],
        'P_det_star': [], 'P_det_opt': [],
        'aoi_star': [], 'aoi_opt': [],
        'success_star': [], 'success_opt': [],
    }
    
    print(f"\n{'c̄/(Bρ)':<8} {'c_min':<7} {'c_max':<7} {'k*':<5} {'k_opt':<6} "
          f"{'Gap':<5} {'Succ*%':<8} {'Succ_opt%':<8}")
    print("-"*75)
    
    for mean_c in mean_costs:
        # Compute c_min, c_max from mean and ratio
        c_min = 2 * mean_c / (1 + spread_ratio)
        c_max = spread_ratio * c_min
        
        cost_params = HeterogeneousCostParams(c_min=c_min, c_max=c_max)
        
        k_star = compute_k_star_heterogeneous(config.N, rho, config.B, cost_params)
        k_opt = compute_k_opt_heterogeneous(config.N, rho, config.B, cost_params)
        
        P_star = analytical_P_det(k_star, rho)
        P_opt = analytical_P_det(k_opt, rho)
        aoi_star = analytical_aoi(k_star, rho)
        aoi_opt = analytical_aoi(k_opt, rho)
        
        # Run dynamic simulations
        success_star = 0
        success_opt = 0
        
        for run in range(n_runs):
            sim = DynamicSimulation(config, k_star, seed=run)
            if sim.run()['success']:
                success_star += 1
            
            sim = DynamicSimulation(config, k_opt, seed=run + 10000)
            if sim.run()['success']:
                success_opt += 1
        
        results['mean_cost'].append(mean_c)
        results['c_norm'].append(mean_c / B_rho)
        results['c_min'].append(c_min)
        results['c_max'].append(c_max)
        results['k_star'].append(k_star)
        results['k_opt'].append(k_opt)
        results['gap'].append(k_opt - k_star)
        results['P_det_star'].append(P_star)
        results['P_det_opt'].append(P_opt)
        results['aoi_star'].append(aoi_star)
        results['aoi_opt'].append(aoi_opt)
        results['success_star'].append(success_star / n_runs)
        results['success_opt'].append(success_opt / n_runs)
        
        print(f"{mean_c/B_rho:<8.2f} {c_min:<7.4f} {c_max:<7.4f} "
              f"{k_star:<5} {k_opt:<6} {k_opt-k_star:<5} "
              f"{100*success_star/n_runs:<8.1f} {100*success_opt/n_runs:<8.1f}")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_all(validation: Dict, sweep_hom: Dict, sweep_het: Dict = None,
                output_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'figures')):
    """Generate all plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    if sweep_het is not None:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. P_det validation (static)
    ax = axes[0, 0]
    k = np.array(validation['k'])
    ax.plot(k, validation['theory'], 'b-', lw=2, label='Theory: $1-(1-ρ)^k$')
    ax.errorbar(k, validation['sim_mean'], yerr=2*np.array(validation['sim_std']),
                fmt='ro', capsize=3, label='Static sim (±2σ)')
    ax.set_xlabel('Active Volunteers k')
    ax.set_ylabel('Detection Probability')
    ax.set_title('P_det Validation (Static i.i.d.)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Validation error
    ax = axes[0, 1]
    colors = ['green' if e < 0.05 else 'orange' if e < 0.10 else 'red' 
              for e in validation['error']]
    ax.bar(k, np.array(validation['error']) * 100, color=colors)
    ax.axhline(5, color='green', ls='--', label='5%')
    ax.axhline(10, color='orange', ls='--', label='10%')
    ax.set_xlabel('Active Volunteers k')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Validation Error')
    ax.legend()
    
    # 3. Participation: k* vs k_opt (homogeneous)
    ax = axes[0, 2]
    c_norm = np.array(sweep_hom['c_norm'])
    ax.plot(c_norm, sweep_hom['k_star'], 'b-o', label='Nash $k^*$')
    ax.plot(c_norm, sweep_hom['k_opt'], 'g-s', label='Optimal $k^{opt}$')
    ax.fill_between(c_norm, sweep_hom['k_star'], sweep_hom['k_opt'], alpha=0.2, color='red',
                    label='Gap')
    ax.axvline(1.0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Active Volunteers')
    ax.set_title('Participation Gap (Homogeneous)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Theoretical AoI (homogeneous)
    ax = axes[1, 0]
    ax.plot(c_norm, sweep_hom['aoi_star'], 'b-o', label='Nash AoI')
    ax.plot(c_norm, sweep_hom['aoi_opt'], 'g-s', label='Optimal AoI')
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Expected AoI (steps)')
    ax.set_title('Theoretical AoI (Homogeneous)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 5. Success rate (homogeneous)
    ax = axes[1, 1]
    ax.plot(c_norm, np.array(sweep_hom['success_star']) * 100, 'b-o', label='Nash')
    ax.plot(c_norm, np.array(sweep_hom['success_opt']) * 100, 'g-s', label='Optimal')
    ax.fill_between(c_norm, 
                    np.array(sweep_hom['success_star']) * 100,
                    np.array(sweep_hom['success_opt']) * 100,
                    alpha=0.2, color='red', label='Efficiency loss')
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Rescue Success Rate (%)')
    ax.set_title('Success Rate (Homogeneous)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    avg_gap_hom = np.mean(sweep_hom['gap'])
    avg_loss_hom = np.mean(np.array(sweep_hom['success_opt']) - np.array(sweep_hom['success_star'])) * 100
    max_loss_hom = np.max(np.array(sweep_hom['success_opt']) - np.array(sweep_hom['success_star'])) * 100
    
    summary = f"""
    SUMMARY - HOMOGENEOUS
    ═══════════════════════════════
    
    Average participation gap: {avg_gap_hom:.1f}
    Average efficiency loss: {avg_loss_hom:.1f}%
    Maximum efficiency loss: {max_loss_hom:.1f}%
    """
    
    if sweep_het is not None:
        avg_gap_het = np.mean(sweep_het['gap'])
        avg_loss_het = np.mean(np.array(sweep_het['success_opt']) - np.array(sweep_het['success_star'])) * 100
        max_loss_het = np.max(np.array(sweep_het['success_opt']) - np.array(sweep_het['success_star'])) * 100
        
        summary += f"""
    SUMMARY - HETEROGENEOUS
    ═══════════════════════════════
    
    Average participation gap: {avg_gap_het:.1f}
    Average efficiency loss: {avg_loss_het:.1f}%
    Maximum efficiency loss: {max_loss_het:.1f}%
    
    COMPARISON
    ═══════════════════════════════
    Gap difference: {avg_gap_het - avg_gap_hom:+.1f}
    Loss difference: {avg_loss_het - avg_loss_hom:+.1f}%
    """
    
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Heterogeneous plots if available
    if sweep_het is not None:
        # 7. Participation (heterogeneous)
        ax = axes[0, 3]
        c_norm_het = np.array(sweep_het['c_norm'])
        ax.plot(c_norm_het, sweep_het['k_star'], 'b-o', label='Nash $k^*$')
        ax.plot(c_norm_het, sweep_het['k_opt'], 'g-s', label='Optimal $k^{opt}$')
        ax.fill_between(c_norm_het, sweep_het['k_star'], sweep_het['k_opt'], 
                        alpha=0.2, color='red', label='Gap')
        ax.set_xlabel('Normalized Mean Cost $\\bar{c}/(Bρ)$')
        ax.set_ylabel('Active Volunteers')
        ax.set_title('Participation Gap (Heterogeneous)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Success rate (heterogeneous)
        ax = axes[1, 3]
        ax.plot(c_norm_het, np.array(sweep_het['success_star']) * 100, 'b-o', label='Nash')
        ax.plot(c_norm_het, np.array(sweep_het['success_opt']) * 100, 'g-s', label='Optimal')
        ax.fill_between(c_norm_het,
                        np.array(sweep_het['success_star']) * 100,
                        np.array(sweep_het['success_opt']) * 100,
                        alpha=0.2, color='red', label='Efficiency loss')
        ax.set_xlabel('Normalized Mean Cost $\\bar{c}/(Bρ)$')
        ax.set_ylabel('Rescue Success Rate (%)')
        ax.set_title('Success Rate (Heterogeneous)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/complete_analysis.png', dpi=150)
    plt.savefig(f'{output_dir}/complete_analysis.pdf')
    print(f"\nSaved: {output_dir}/complete_analysis.png")


# =============================================================================
# MAIN
# =============================================================================

def main(use_heterogeneous: bool = True):
    """
    Main function.
    
    Parameters
    ----------
    use_heterogeneous : bool
        If True, run both homogeneous and heterogeneous analysis.
        If False, run only homogeneous (original behavior).
    """
    print("="*60)
    print("EMERGENCY CROWD-FINDING SIMULATION")
    print("="*60)
    
    config = Config(L=500, N=100, R=30, B=10)
    
    print(f"\nConfiguration:")
    print(f"  L = {config.L}m, N = {config.N}, R = {config.R}m")
    print(f"  ρ = {config.rho:.6f}")
    print(f"  Bρ = {config.B_rho:.4f}")
    print(f"  NBρ = {config.NB_rho:.4f}")
    print(f"  Use heterogeneous model: {use_heterogeneous}")
    
    # 1. Validate P_det with STATIC simulation
    print("\n" + "="*60)
    print("STEP 1: Validate P_det (static i.i.d. simulation)")
    print("="*60)
    
    k_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    validation = validate_P_det_static(config, k_values, n_runs=100, n_steps=2000)
    
    # 2. Run homogeneous cost sweep
    print("\n" + "="*60)
    print("STEP 2a: Cost sweep - Homogeneous model")
    print("="*60)
    
    sweep_hom = run_informative_sweep_homogeneous(config, n_runs=30)
    
    # 3. Run heterogeneous cost sweep (if enabled)
    sweep_het = None
    if use_heterogeneous:
        print("\n" + "="*60)
        print("STEP 2b: Cost sweep - Heterogeneous model")
        print("="*60)
        
        sweep_het = run_informative_sweep_heterogeneous(config, n_runs=30, spread_ratio=2.0)
    
    # 4. Plot
    print("\n" + "="*60)
    print("STEP 3: Generate plots")
    print("="*60)
    
    plot_all(validation, sweep_hom, sweep_het)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--homogeneous-only', action='store_true',
                        help='Run only homogeneous model (original behavior)')
    args = parser.parse_args()
    
    main(use_heterogeneous=not args.homogeneous_only)