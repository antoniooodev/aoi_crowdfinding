"""Experiment 06: Heterogeneous cost model analysis.

Compares homogeneous and heterogeneous cost models across a heterogeneity sweep, including equilibrium participation, welfare/PoA, and Stackelberg incentives.

Outputs:
    results/data/exp06_heterogeneous_analysis.csv
    results/figures/exp06_heterogeneity_comparison.pdf
    results/figures/exp06_poa_vs_heterogeneity.pdf
    results/figures/exp06_incentive_comparison.pdf

Run:
    python -m experiments.exp06_heterogeneous_analysis
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    HeterogeneousCostParams,
    compute_heterogeneity_sweep,
    PhysicalParams,
)
from src.game import (
    find_nash_equilibrium,
    find_social_optimum,
    price_of_anarchy,
    analyze_equilibrium,
    find_nash_heterogeneous,
    find_social_optimum_heterogeneous_expected,
    price_of_anarchy_heterogeneous,
    analyze_equilibrium_heterogeneous,
    sweep_heterogeneity,
    compare_homogeneous_heterogeneous,
)
from src.stackelberg import (
    analyze_stackelberg,
    analyze_stackelberg_heterogeneous,
    compare_incentives_homogeneous_heterogeneous,
)
from src.simulation import (
    validate_heterogeneous_model,
    sweep_heterogeneity_validation,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

FIGURES_DIR = 'results/figures'
DATA_DIR = 'results/data'

# Default parameters
DEFAULT_L = 500.0
DEFAULT_R = 30.0
DEFAULT_N = 100
DEFAULT_B = 10.0


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# EXPERIMENT 1: Effect of Heterogeneity on Equilibrium
# =============================================================================

def exp1_heterogeneity_effect(N: int = DEFAULT_N, R: float = DEFAULT_R, 
                               L: float = DEFAULT_L, B: float = DEFAULT_B):
    """
    Analyze how heterogeneity ratio affects k*, k_opt, and PoA.
    
    Fix mean cost, vary spread ratio from 1 (homogeneous) to 5 (high heterogeneity).
    """
    print("\n" + "="*60)
    print("EXP 1: Effect of Heterogeneity on Equilibrium")
    print("="*60)
    
    rho = np.pi * R**2 / L**2
    B_rho = B * rho
    
    # Fix mean cost at 0.5 * Bρ (middle of informative range)
    mean_cost = 0.5 * B_rho
    
    # Vary spread ratio
    spread_ratios = np.linspace(1.0, 5.0, 20)
    
    print(f"Parameters: N={N}, ρ={rho:.6f}, B={B}, mean_cost={mean_cost:.4f}")
    print(f"Spread ratios: {spread_ratios[0]:.1f} to {spread_ratios[-1]:.1f}")
    
    # Run sweep
    results = sweep_heterogeneity(N, rho, B, mean_cost, spread_ratios)
    
    # Add homogeneous baseline
    hom_analysis = analyze_equilibrium(N, R, L, B, mean_cost)
    
    print(f"\nHomogeneous baseline (c={mean_cost:.4f}):")
    print(f"  k* = {hom_analysis['k_star']}, k_opt = {hom_analysis['k_opt']}")
    print(f"  Gap = {hom_analysis['participation_gap']}, PoA = {hom_analysis['price_of_anarchy']:.3f}")
    
    print(f"\nHeterogeneous (spread_ratio=3.0):")
    idx = np.argmin(np.abs(spread_ratios - 3.0))
    print(f"  k* = {results['k_star'][idx]}, k_opt = {results['k_opt'][idx]}")
    print(f"  Gap = {results['gap'][idx]}, PoA = {results['poa'][idx]:.3f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. k* and k_opt vs heterogeneity
    ax = axes[0, 0]
    ax.plot(spread_ratios, results['k_star'], 'b-o', label='Nash $k^*$', markersize=4)
    ax.plot(spread_ratios, results['k_opt'], 'g-s', label='Optimal $k^{opt}$', markersize=4)
    ax.axhline(hom_analysis['k_star'], color='blue', ls='--', alpha=0.5, label='Hom. $k^*$')
    ax.axhline(hom_analysis['k_opt'], color='green', ls='--', alpha=0.5, label='Hom. $k^{opt}$')
    ax.set_xlabel('Heterogeneity Ratio $c_{max}/c_{min}$')
    ax.set_ylabel('Active Volunteers')
    ax.set_title('Participation vs Heterogeneity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Participation gap vs heterogeneity
    ax = axes[0, 1]
    ax.plot(spread_ratios, results['gap'], 'r-o', markersize=4)
    ax.axhline(hom_analysis['participation_gap'], color='red', ls='--', alpha=0.5,
               label=f'Homogeneous: {hom_analysis["participation_gap"]}')
    ax.set_xlabel('Heterogeneity Ratio $c_{max}/c_{min}$')
    ax.set_ylabel('Participation Gap $k^{opt} - k^*$')
    ax.set_title('Participation Gap vs Heterogeneity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. PoA vs heterogeneity
    ax = axes[1, 0]
    ax.plot(spread_ratios, results['poa'], 'm-o', markersize=4)
    ax.axhline(hom_analysis['price_of_anarchy'], color='magenta', ls='--', alpha=0.5,
               label=f'Homogeneous: {hom_analysis["price_of_anarchy"]:.2f}')
    ax.set_xlabel('Heterogeneity Ratio $c_{max}/c_{min}$')
    ax.set_ylabel('Price of Anarchy')
    ax.set_title('PoA vs Heterogeneity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Welfare comparison
    ax = axes[1, 1]
    ax.plot(spread_ratios, results['welfare_star'], 'b-o', label='Nash $W^*$', markersize=4)
    ax.plot(spread_ratios, results['welfare_opt'], 'g-s', label='Optimal $W^{opt}$', markersize=4)
    ax.axhline(hom_analysis['welfare_star'], color='blue', ls='--', alpha=0.5)
    ax.axhline(hom_analysis['welfare_opt'], color='green', ls='--', alpha=0.5)
    ax.set_xlabel('Heterogeneity Ratio $c_{max}/c_{min}$')
    ax.set_ylabel('Social Welfare')
    ax.set_title('Welfare vs Heterogeneity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/exp06_heterogeneity_effect.pdf')
    plt.savefig(f'{FIGURES_DIR}/exp06_heterogeneity_effect.png', dpi=150)
    print(f"\nSaved: {FIGURES_DIR}/exp06_heterogeneity_effect.pdf")
    
    return results, hom_analysis


# =============================================================================
# EXPERIMENT 2: Homogeneous vs Heterogeneous Cost Sweep
# =============================================================================

def exp2_cost_sweep_comparison(N: int = DEFAULT_N, R: float = DEFAULT_R,
                                L: float = DEFAULT_L, B: float = DEFAULT_B):
    """
    Compare homogeneous and heterogeneous models across cost sweep.
    """
    print("\n" + "="*60)
    print("EXP 2: Homogeneous vs Heterogeneous Cost Sweep")
    print("="*60)
    
    rho = np.pi * R**2 / L**2
    B_rho = B * rho
    
    # Cost sweep
    c_values = np.linspace(0.1 * B_rho, 0.9 * B_rho, 15)
    spread_ratio = 2.0
    
    print(f"Parameters: N={N}, ρ={rho:.6f}, B={B}")
    print(f"Cost range: {c_values[0]:.4f} to {c_values[-1]:.4f}")
    print(f"Heterogeneity ratio: {spread_ratio}")
    
    results = {
        'c': [], 'c_norm': [],
        'k_star_hom': [], 'k_opt_hom': [], 'gap_hom': [], 'poa_hom': [],
        'k_star_het': [], 'k_opt_het': [], 'gap_het': [], 'poa_het': [],
    }
    
    for c in c_values:
        # Homogeneous
        hom = analyze_equilibrium(N, R, L, B, c)
        
        # Heterogeneous with same mean
        c_min = 2 * c / (1 + spread_ratio)
        c_max = spread_ratio * c_min
        cost_params = HeterogeneousCostParams(c_min=c_min, c_max=c_max)
        het = analyze_equilibrium_heterogeneous(N, rho, B, cost_params)
        
        results['c'].append(c)
        results['c_norm'].append(c / B_rho)
        results['k_star_hom'].append(hom['k_star'])
        results['k_opt_hom'].append(hom['k_opt'])
        results['gap_hom'].append(hom['participation_gap'])
        results['poa_hom'].append(hom['price_of_anarchy'])
        results['k_star_het'].append(het['k_star'])
        results['k_opt_het'].append(het['k_opt'])
        results['gap_het'].append(het['participation_gap'])
        results['poa_het'].append(het['price_of_anarchy'])
    
    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    c_norm = results['c_norm']
    
    # 1. k* comparison
    ax = axes[0, 0]
    ax.plot(c_norm, results['k_star_hom'], 'b-o', label='Homogeneous', markersize=4)
    ax.plot(c_norm, results['k_star_het'], 'r-s', label='Heterogeneous', markersize=4)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Nash Equilibrium $k^*$')
    ax.set_title('Nash Participation: Hom vs Het')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. k_opt comparison
    ax = axes[0, 1]
    ax.plot(c_norm, results['k_opt_hom'], 'b-o', label='Homogeneous', markersize=4)
    ax.plot(c_norm, results['k_opt_het'], 'r-s', label='Heterogeneous', markersize=4)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Social Optimum $k^{opt}$')
    ax.set_title('Optimal Participation: Hom vs Het')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Gap comparison
    ax = axes[1, 0]
    ax.plot(c_norm, results['gap_hom'], 'b-o', label='Homogeneous', markersize=4)
    ax.plot(c_norm, results['gap_het'], 'r-s', label='Heterogeneous', markersize=4)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Participation Gap')
    ax.set_title('Participation Gap: Hom vs Het')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. PoA comparison
    ax = axes[1, 1]
    poa_hom_clipped = np.clip(results['poa_hom'], 0, 10)
    poa_het_clipped = np.clip(results['poa_het'], 0, 10)
    ax.plot(c_norm, poa_hom_clipped, 'b-o', label='Homogeneous', markersize=4)
    ax.plot(c_norm, poa_het_clipped, 'r-s', label='Heterogeneous', markersize=4)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Price of Anarchy')
    ax.set_title('PoA: Hom vs Het (clipped at 10)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/exp06_cost_sweep_comparison.pdf')
    plt.savefig(f'{FIGURES_DIR}/exp06_cost_sweep_comparison.png', dpi=150)
    print(f"\nSaved: {FIGURES_DIR}/exp06_cost_sweep_comparison.pdf")
    
    return results


# =============================================================================
# EXPERIMENT 3: Stackelberg Incentive Comparison
# =============================================================================

def exp3_incentive_comparison(N: int = DEFAULT_N, R: float = DEFAULT_R,
                               L: float = DEFAULT_L, B: float = DEFAULT_B):
    """
    Compare Stackelberg incentives for homogeneous vs heterogeneous.
    """
    print("\n" + "="*60)
    print("EXP 3: Stackelberg Incentive Comparison")
    print("="*60)
    
    rho = np.pi * R**2 / L**2
    B_rho = B * rho
    
    # Cost values
    c_values = np.linspace(0.2 * B_rho, 0.8 * B_rho, 10)
    spread_ratio = 2.0
    
    results = {
        'c': [], 'c_norm': [],
        'p_star_hom': [], 'budget_hom': [], 'k_stack_hom': [],
        'p_star_het': [], 'budget_het': [], 'k_stack_het': [],
        'efficiency_gain_hom': [], 'efficiency_gain_het': [],
    }
    
    print(f"\n{'c/(Bρ)':<8} {'p*_hom':<10} {'p*_het':<10} {'Budget_hom':<12} {'Budget_het':<12}")
    print("-"*60)
    
    for c in c_values:
        comparison = compare_incentives_homogeneous_heterogeneous(N, R, L, B, c, spread_ratio)
        hom = comparison['homogeneous']
        het = comparison['heterogeneous']
        
        results['c'].append(c)
        results['c_norm'].append(c / B_rho)
        results['p_star_hom'].append(hom['p_stackelberg'])
        results['budget_hom'].append(hom['incentive_cost'])
        results['k_stack_hom'].append(hom['k_stackelberg'])
        results['p_star_het'].append(het['p_stackelberg'])
        results['budget_het'].append(het['incentive_cost'])
        results['k_stack_het'].append(het['k_stackelberg'])
        results['efficiency_gain_hom'].append(hom['efficiency_gain'])
        results['efficiency_gain_het'].append(het['efficiency_gain'])
        
        print(f"{c/B_rho:<8.2f} {hom['p_stackelberg']:<10.4f} {het['p_stackelberg']:<10.4f} "
              f"{hom['incentive_cost']:<12.4f} {het['incentive_cost']:<12.4f}")
    
    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    c_norm = results['c_norm']
    
    # 1. Optimal incentive
    ax = axes[0, 0]
    ax.plot(c_norm, results['p_star_hom'], 'b-o', label='Homogeneous', markersize=5)
    ax.plot(c_norm, results['p_star_het'], 'r-s', label='Heterogeneous', markersize=5)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Optimal Incentive $p^*$')
    ax.set_title('Optimal Per-Volunteer Incentive')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Total budget
    ax = axes[0, 1]
    ax.plot(c_norm, results['budget_hom'], 'b-o', label='Homogeneous', markersize=5)
    ax.plot(c_norm, results['budget_het'], 'r-s', label='Heterogeneous', markersize=5)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Total Budget $p^* \\cdot k$')
    ax.set_title('Total Incentive Budget')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Induced participation
    ax = axes[1, 0]
    ax.plot(c_norm, results['k_stack_hom'], 'b-o', label='Homogeneous', markersize=5)
    ax.plot(c_norm, results['k_stack_het'], 'r-s', label='Heterogeneous', markersize=5)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Stackelberg Participation $k_{stack}$')
    ax.set_title('Induced Participation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Efficiency gain
    ax = axes[1, 1]
    ax.plot(c_norm, np.array(results['efficiency_gain_hom']) * 100, 'b-o', 
            label='Homogeneous', markersize=5)
    ax.plot(c_norm, np.array(results['efficiency_gain_het']) * 100, 'r-s', 
            label='Heterogeneous', markersize=5)
    ax.set_xlabel('Normalized Cost $c/(Bρ)$')
    ax.set_ylabel('Efficiency Gain (%)')
    ax.set_title('Welfare Improvement from Incentives')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/exp06_incentive_comparison.pdf')
    plt.savefig(f'{FIGURES_DIR}/exp06_incentive_comparison.png', dpi=150)
    print(f"\nSaved: {FIGURES_DIR}/exp06_incentive_comparison.pdf")
    
    return results


# =============================================================================
# EXPERIMENT 4: Monte Carlo Validation
# =============================================================================

def exp4_monte_carlo_validation(N: int = DEFAULT_N, R: float = DEFAULT_R,
                                 L: float = DEFAULT_L, B: float = DEFAULT_B,
                                 n_runs: int = 500):
    """
    Validate heterogeneous model predictions via Monte Carlo.
    """
    print("\n" + "="*60)
    print("EXP 4: Monte Carlo Validation")
    print("="*60)
    
    rho = np.pi * R**2 / L**2
    B_rho = B * rho
    
    # Test configuration
    mean_cost = 0.5 * B_rho
    spread_ratios = np.array([1.5, 2.0, 3.0, 4.0])
    
    print(f"Parameters: N={N}, ρ={rho:.6f}, B={B}")
    print(f"Mean cost: {mean_cost:.4f}, Spread ratios: {spread_ratios}")
    print(f"Monte Carlo runs: {n_runs}")
    print()
    
    results = sweep_heterogeneity_validation(N, rho, B, mean_cost, spread_ratios.tolist(), n_runs)
    
    # Print results
    print(f"{'Ratio':<8} {'k*_ana':<8} {'k*_sim':<8} {'Error%':<8} "
          f"{'k_opt_ana':<10} {'k_opt_sim':<10} {'Error%':<8}")
    print("-"*70)
    
    for i, ratio in enumerate(spread_ratios):
        print(f"{ratio:<8.1f} {results['k_star_analytical'][i]:<8} "
              f"{results['k_star_simulated'][i]:<8.2f} {results['k_star_error'][i]*100:<8.2f} "
              f"{results['k_opt_analytical'][i]:<10} {results['k_opt_simulated'][i]:<10.2f} "
              f"{results['k_opt_error'][i]*100:<8.2f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. k* validation
    ax = axes[0]
    ax.bar(spread_ratios - 0.15, results['k_star_analytical'], 0.3, 
           label='Analytical', color='blue', alpha=0.7)
    ax.bar(spread_ratios + 0.15, results['k_star_simulated'], 0.3,
           label='Simulated', color='red', alpha=0.7)
    ax.set_xlabel('Heterogeneity Ratio')
    ax.set_ylabel('Nash Equilibrium $k^*$')
    ax.set_title('Nash Equilibrium Validation')
    ax.legend()
    ax.set_xticks(spread_ratios)
    
    # 2. k_opt validation
    ax = axes[1]
    ax.bar(spread_ratios - 0.15, results['k_opt_analytical'], 0.3,
           label='Analytical', color='blue', alpha=0.7)
    ax.bar(spread_ratios + 0.15, results['k_opt_simulated'], 0.3,
           label='Simulated', color='red', alpha=0.7)
    ax.set_xlabel('Heterogeneity Ratio')
    ax.set_ylabel('Social Optimum $k^{opt}$')
    ax.set_title('Social Optimum Validation')
    ax.legend()
    ax.set_xticks(spread_ratios)
    
    # 3. PoA validation
    ax = axes[2]
    ax.bar(spread_ratios - 0.15, results['poa_analytical'], 0.3,
           label='Analytical', color='blue', alpha=0.7)
    ax.bar(spread_ratios + 0.15, results['poa_simulated'], 0.3,
           label='Simulated', color='red', alpha=0.7)
    ax.set_xlabel('Heterogeneity Ratio')
    ax.set_ylabel('Price of Anarchy')
    ax.set_title('PoA Validation')
    ax.legend()
    ax.set_xticks(spread_ratios)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/exp06_mc_validation.pdf')
    plt.savefig(f'{FIGURES_DIR}/exp06_mc_validation.png', dpi=150)
    print(f"\nSaved: {FIGURES_DIR}/exp06_mc_validation.pdf")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def run():
    """Run all heterogeneous model experiments."""
    print("="*60)
    print("EXP06: HETEROGENEOUS COST MODEL ANALYSIS")
    print("="*60)
    
    ensure_dirs()
    
    # Run experiments
    results1, hom_baseline = exp1_heterogeneity_effect()
    results2 = exp2_cost_sweep_comparison()
    results3 = exp3_incentive_comparison()
    results4 = exp4_monte_carlo_validation(n_runs=200)  # Reduced for speed
    
    # Save summary data
    summary = {
        'experiment': ['heterogeneity_effect', 'cost_sweep', 'incentive', 'validation'],
        'description': [
            'Effect of spread ratio on equilibrium',
            'Hom vs Het across cost range',
            'Stackelberg incentive comparison',
            'Monte Carlo validation'
        ],
        'status': ['complete'] * 4
    }
    df = pd.DataFrame(summary)
    df.to_csv(f'{DATA_DIR}/exp06_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("EXP06 COMPLETE")
    print("="*60)
    print(f"Figures saved to: {FIGURES_DIR}/exp06_*.pdf")
    print(f"Data saved to: {DATA_DIR}/exp06_*.csv")


if __name__ == '__main__':
    run()