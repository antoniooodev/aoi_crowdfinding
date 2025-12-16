"""
emergency_simulation_v3.py - Final Corrected Simulation

TWO MODES:
1. STATIC: i.i.d. positions each step → validates P_det formula exactly
2. DYNAMIC: agent movement → shows qualitative under-participation effect

Key insight: The agent-based simulation is NOT meant to validate the exact formula,
but to demonstrate the qualitative phenomenon of under-participation.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum
import os


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
    
    @property
    def rho(self) -> float:
        return np.pi * self.R**2 / self.L**2
    
    @property
    def B_rho(self) -> float:
        return self.B * self.rho
    
    @property
    def NB_rho(self) -> float:
        return self.N * self.B * self.rho


def compute_k_star(N: int, rho: float, B: float, c: float) -> int:
    """Nash equilibrium."""
    if c >= B * rho:
        return 0
    if c <= 0:
        return N
    k = 1 + np.log(c / (B * rho)) / np.log(1 - rho)
    return max(0, min(N, int(np.floor(k))))


def compute_k_opt(N: int, rho: float, B: float, c: float) -> int:
    """Social optimum."""
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
        
        # Move target
        if self.rng.random() < 0.02:
            angle = self.rng.uniform(0, 2*np.pi)
            self.target_vx = np.cos(angle) * self.config.target_speed
            self.target_vy = np.sin(angle) * self.config.target_speed
        
        margin = self.config.R
        new_x = self.target_x + self.target_vx
        new_y = self.target_y + self.target_vy
        
        if new_x < margin or new_x > self.config.L - margin:
            self.target_vx *= -1
        if new_y < margin or new_y > self.config.L - margin:
            self.target_vy *= -1
        
        self.target_x = np.clip(new_x, margin, self.config.L - margin)
        self.target_y = np.clip(new_y, margin, self.config.L - margin)
        
        # Check detection
        active_x = self.vol_x[self.vol_active]
        active_y = self.vol_y[self.vol_active]
        if len(active_x) > 0:
            distances = np.sqrt((active_x - self.target_x)**2 + 
                               (active_y - self.target_y)**2)
            detected = np.any(distances <= self.config.R)
        else:
            detected = False
        
        # Update AoI
        if detected:
            self.aoi = 0
            self.detections += 1
            if not self.rescue_started:
                self.rescue_started = True
                self.rescue_start_time = self.time
        else:
            self.aoi += 1
        
        self.aoi_history.append(self.aoi)
        
        # Check rescue completion
        if self.rescue_started:
            if self.aoi > self.config.aoi_threshold:
                # Failed - AoI exceeded during rescue
                self.rescue_started = False
                self.rescue_start_time = None
            elif self.time - self.rescue_start_time >= self.config.response_time:
                self.success = True
        
        self.time += 1
        return self.success
    
    def run(self) -> Dict:
        """Run to completion or timeout."""
        while self.time < self.config.T_max and not self.success:
            self.step()
        
        return {
            'k': self.k,
            'success': self.success,
            'time': self.time,
            'detections': self.detections,
            'mean_aoi': np.mean(self.aoi_history) if self.aoi_history else np.inf,
            'max_aoi': np.max(self.aoi_history) if self.aoi_history else np.inf,
        }


# =============================================================================
# INFORMATIVE COST SWEEP
# =============================================================================

def run_informative_sweep(config: Config, n_runs: int = 50) -> Dict:
    """
    Sweep cost in region where BOTH k* and k_opt vary.
    
    For k* to vary: c ∈ (0, Bρ)
    For k_opt to vary: c ∈ (0, NBρ)
    
    We want c where k* ∈ (0, N) and k_opt ∈ (0, N).
    """
    print("\n" + "="*60)
    print("INFORMATIVE COST SWEEP")
    print("="*60)
    
    rho = config.rho
    B_rho = config.B_rho
    NB_rho = config.NB_rho
    
    print(f"Bρ = {B_rho:.4f} (threshold for k* > 0)")
    print(f"NBρ = {NB_rho:.4f} (threshold for k_opt > 0)")
    
    # Sweep c from 0.1*Bρ to 0.95*Bρ
    # This gives k* from ~N down to ~0
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
            # Nash
            sim = DynamicSimulation(config, k_star, seed=run)
            if sim.run()['success']:
                success_star += 1
            
            # Optimal
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
# VISUALIZATION
# =============================================================================

def plot_all(validation: Dict, sweep: Dict, output_dir: str = 'results/figures'):
    """Generate all plots."""
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 3. Participation: k* vs k_opt
    ax = axes[0, 2]
    c_norm = np.array(sweep['c_norm'])
    ax.plot(c_norm, sweep['k_star'], 'b-o', label='Nash $k^*$')
    ax.plot(c_norm, sweep['k_opt'], 'g-s', label='Optimal $k^{opt}$')
    ax.fill_between(c_norm, sweep['k_star'], sweep['k_opt'], alpha=0.2, color='red',
                    label='Under-participation gap')
    ax.axvline(1.0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Active Volunteers')
    ax.set_title('Participation Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Theoretical AoI
    ax = axes[1, 0]
    ax.plot(c_norm, sweep['aoi_star'], 'b-o', label='Nash AoI')
    ax.plot(c_norm, sweep['aoi_opt'], 'g-s', label='Optimal AoI')
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Expected AoI (steps)')
    ax.set_title('Theoretical AoI Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 5. Success rate (dynamic simulation)
    ax = axes[1, 1]
    ax.plot(c_norm, np.array(sweep['success_star']) * 100, 'b-o', label='Nash')
    ax.plot(c_norm, np.array(sweep['success_opt']) * 100, 'g-s', label='Optimal')
    ax.fill_between(c_norm, 
                    np.array(sweep['success_star']) * 100,
                    np.array(sweep['success_opt']) * 100,
                    alpha=0.2, color='red', label='Efficiency loss')
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Rescue Success Rate (%)')
    ax.set_title('Success Rate (Dynamic Simulation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    avg_gap = np.mean(sweep['gap'])
    avg_loss = np.mean(np.array(sweep['success_opt']) - np.array(sweep['success_star'])) * 100
    max_loss = np.max(np.array(sweep['success_opt']) - np.array(sweep['success_star'])) * 100
    
    summary = f"""
    SUMMARY
    ═══════════════════════════════
    
    Average participation gap: {avg_gap:.1f}
    
    Average efficiency loss: {avg_loss:.1f}%
    Maximum efficiency loss: {max_loss:.1f}%
    
    Key insight:
    As cost increases toward Bρ,
    Nash participation drops,
    while social optimum remains high.
    
    This causes increasing efficiency loss
    in rescue success rate.
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/complete_analysis.png', dpi=150)
    plt.savefig(f'{output_dir}/complete_analysis.pdf')
    print(f"\nSaved: {output_dir}/complete_analysis.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("FINAL CORRECTED SIMULATION")
    print("="*60)
    
    config = Config(L=500, N=100, R=30, B=10)
    
    print(f"\nConfiguration:")
    print(f"  L = {config.L}m, N = {config.N}, R = {config.R}m")
    print(f"  ρ = {config.rho:.6f}")
    print(f"  Bρ = {config.B_rho:.4f}")
    print(f"  NBρ = {config.NB_rho:.4f}")
    
    # 1. Validate P_det with STATIC simulation
    print("\n" + "="*60)
    print("STEP 1: Validate P_det (static i.i.d. simulation)")
    print("="*60)
    
    k_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    validation = validate_P_det_static(config, k_values, n_runs=100, n_steps=2000)
    
    # 2. Run informative cost sweep with DYNAMIC simulation
    print("\n" + "="*60)
    print("STEP 2: Cost sweep (dynamic agent simulation)")
    print("="*60)
    
    sweep = run_informative_sweep(config, n_runs=30)
    
    # 3. Plot
    print("\n" + "="*60)
    print("STEP 3: Generate plots")
    print("="*60)
    
    plot_all(validation, sweep)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
