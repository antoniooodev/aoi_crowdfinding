"""
run_experiments.py - Run all experiments with corrected parameters

This script runs the complete experimental suite with the corrections applied:
1. Interior target sampling (matches analytical theory)
2. Adaptive cost ranges (based on Bρ and NBρ thresholds)
3. Proper PoA visualization (masks infinite values)
4. Robust Stackelberg incentives (epsilon fix)
"""

import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    compute_informative_cost_range,
    compute_normalized_cost_range,
    print_thresholds,
    suggest_parameters,
)
from src.spatial import compute_rho, compute_thresholds
from src.aoi import expected_aoi_from_k, aoi_vectorized
from src.game import (
    find_nash_equilibrium,
    find_social_optimum,
    social_welfare,
    price_of_anarchy,
    analyze_equilibrium,
    sweep_cost,
)
from src.simulation import (
    MonteCarloSimulation,
    run_parameter_sweep,
    validate_analytical,
)
from src.stackelberg import (
    analyze_stackelberg,
    incentive_sensitivity,
)
from src.visualization import (
    plot_equilibrium_vs_cost,
    plot_aoi_vs_k,
    plot_poa_heatmap,
    plot_poa_vs_cost,
    plot_stackelberg_incentive,
    plot_welfare_comparison,
    plot_participation_gap,
    plot_spatial_snapshot,
    plot_validation_results,
    save_figure,
)

from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base parameters
N = 100
L = 100.0
R = 10.0
B = 10.0

# Derived thresholds
rho = compute_rho(R, L)
B_rho = B * rho
NB_rho = N * B * rho

# Output directories
REPO_ROOT = Path(__file__).resolve().parents[1]  
RESULTS_DIR = REPO_ROOT / "results" / "data"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title: str):
    """Print section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)


# =============================================================================
# EXPERIMENT 1: VALIDATION (Theory vs Simulation)
# =============================================================================

def run_validation():
    """Validate analytical formulas against Monte Carlo simulation."""
    print_header("EXPERIMENT 1: VALIDATION")
    
    print(f"\nParameters: N={N}, L={L}, R={R}, ρ={rho:.6f}")
    print(f"Using interior target sampling: [R, L-R]² = [{R}, {L-R}]²")
    
    k_values = list(range(1, 51, 2))  # k = 1, 3, 5, ..., 49
    
    print(f"\nRunning validation for k ∈ {{{k_values[0]}, ..., {k_values[-1]}}}...")
    
    validation = validate_analytical(
        L=L, R=R, k_values=k_values,
        T=10000, n_runs=500,
        tolerance=0.05,
        interior_target=True  # CORRECTION: Use interior sampling
    )
    
    # Save results
    df = pd.DataFrame({
        'k': validation['k'],
        'analytical_aoi': validation['analytical_aoi'],
        'simulated_aoi': validation['simulated_aoi'],
        'simulated_std': validation['simulated_aoi_std'],
        'relative_error': validation['relative_error'],
        'within_tolerance': validation['within_tolerance'],
        'within_3sigma': validation['within_3sigma'],
    })
    df.to_csv(f'{RESULTS_DIR}/validation_results.csv', index=False)
    print(f"\nSaved: {RESULTS_DIR}/validation_results.csv")
    
    # Plot
    plot_validation_results(
        validation['k'],
        validation['analytical_aoi'],
        validation['simulated_aoi'],
        validation['simulated_aoi_std'],
        validation['within_tolerance'],
        save=True,
        output_dir=FIGURES_DIR
    )
    
    # Summary
    n_passed = np.sum(validation['within_tolerance'])
    n_total = len(validation['k'])
    print(f"\nValidation: {n_passed}/{n_total} passed (tolerance=5%)")
    print(f"All within 3σ: {validation['all_within_3sigma']}")
    
    return validation


# =============================================================================
# EXPERIMENT 2: EQUILIBRIUM ANALYSIS
# =============================================================================

def run_equilibrium_analysis():
    """Analyze Nash equilibrium and social optimum vs cost."""
    print_header("EXPERIMENT 2: EQUILIBRIUM ANALYSIS")
    
    print(f"\nThresholds:")
    print(f"  Bρ = {B_rho:.4f}  (max c for k* > 0)")
    print(f"  NBρ = {NB_rho:.4f}  (max c for k_opt > 0)")
    
    # CORRECTION: Use adaptive cost range around thresholds
    c_values, c_normalized = compute_normalized_cost_range(
        N, R, L, B,
        c_normalized_range=(0.01, 2.0),  # 0.01 to 2× threshold
        n_points=50,
        normalize_by='B_rho'
    )
    
    print(f"\nSweeping c ∈ [{c_values[0]:.4f}, {c_values[-1]:.4f}]")
    print(f"Normalized: c/(Bρ) ∈ [{c_normalized[0]:.2f}, {c_normalized[-1]:.2f}]")
    
    # Run sweep
    results = sweep_cost(N, R, L, B, c_values)
    results['c_normalized'] = c_normalized
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f'{RESULTS_DIR}/equilibrium_analysis.csv', index=False)
    print(f"\nSaved: {RESULTS_DIR}/equilibrium_analysis.csv")
    
    # Plot
    params = {'N': N, 'R': R, 'L': L, 'B': B, 'rho': rho}
    
    plot_equilibrium_vs_cost(
        c_values, results['k_ne'], results['k_opt'],
        params=params, c_normalized=c_normalized,
        save=True, output_dir=FIGURES_DIR
    )
    
    plot_participation_gap(
        c_values, results['k_ne'], results['k_opt'],
        params=params, c_normalized=c_normalized,
        save=True, output_dir=FIGURES_DIR
    )
    
    # Summary
    print(f"\nResults at c/(Bρ) = 0.5 (mid-range):")
    mid_idx = len(c_values) // 4
    print(f"  k* = {results['k_ne'][mid_idx]}")
    print(f"  k_opt = {results['k_opt'][mid_idx]}")
    print(f"  Gap = {results['gap'][mid_idx]}")
    
    return results


# =============================================================================
# EXPERIMENT 3: PRICE OF ANARCHY
# =============================================================================

def run_poa_analysis():
    """Analyze Price of Anarchy."""
    print_header("EXPERIMENT 3: PRICE OF ANARCHY")
    
    # CORRECTION: Restrict cost range to where k* > 0 (c < Bρ)
    # This ensures finite PoA values
    c_values, c_normalized = compute_normalized_cost_range(
        N, R, L, B,
        c_normalized_range=(0.01, 0.95),  # Stay below threshold
        n_points=30,
        normalize_by='B_rho'
    )
    
    print(f"\nRestricting to c < Bρ for finite PoA")
    print(f"c ∈ [{c_values[0]:.4f}, {c_values[-1]:.4f}] (all < {B_rho:.4f})")
    
    # PoA vs cost (single N)
    results = sweep_cost(N, R, L, B, c_values)
    
    params = {'N': N, 'R': R, 'L': L, 'B': B, 'rho': rho}
    plot_poa_vs_cost(
        c_values, results['poa'], params=params,
        c_normalized=c_normalized, show_threshold=True,
        save=True, output_dir=FIGURES_DIR
    )
    
    # PoA heatmap over (N, c) grid
    print("\nComputing PoA heatmap...")
    N_values = np.array([50, 75, 100, 150, 200])
    c_grid = np.linspace(B_rho * 0.1, B_rho * 0.9, 20)
    
    poa_matrix = np.zeros((len(N_values), len(c_grid)))
    
    for i, N_val in enumerate(N_values):
        for j, c_val in enumerate(c_grid):
            poa_matrix[i, j] = price_of_anarchy(N_val, R, L, B, c_val)
    
    # CORRECTION: Use mask_infinite=True to handle edge cases
    plot_poa_heatmap(
        N_values, c_grid, poa_matrix,
        params=params, mask_infinite=True,
        save=True, output_dir=FIGURES_DIR
    )
    
    # Save
    df = pd.DataFrame(results)
    df['c_normalized'] = c_normalized
    df.to_csv(f'{RESULTS_DIR}/poa_analysis.csv', index=False)
    
    np.savez(f'{RESULTS_DIR}/poa_heatmap.npz',
             N_values=N_values, c_values=c_grid, poa_matrix=poa_matrix)
    
    print(f"\nPoA range: [{np.min(results['poa']):.3f}, {np.max(results['poa']):.3f}]")
    
    return results, poa_matrix


# =============================================================================
# EXPERIMENT 4: STACKELBERG ANALYSIS
# =============================================================================

def run_stackelberg_analysis():
    """Analyze Stackelberg game with platform incentives."""
    print_header("EXPERIMENT 4: STACKELBERG ANALYSIS")
    
    # Use wider cost range for Stackelberg (can have k_opt > 0 even if k* = 0)
    c_values, c_normalized = compute_normalized_cost_range(
        N, R, L, B,
        c_normalized_range=(0.1, 3.0),
        n_points=40,
        normalize_by='B_rho'
    )
    
    print(f"\nAnalyzing Stackelberg with robust incentive computation...")
    
    # CORRECTION: use_robust=True for proper incentive calculation
    results = incentive_sensitivity(N, R, L, B, c_values, use_robust=True)
    results['c_normalized'] = c_normalized
    
    # Save
    df = pd.DataFrame(results)
    df.to_csv(f'{RESULTS_DIR}/stackelberg_analysis.csv', index=False)
    print(f"\nSaved: {RESULTS_DIR}/stackelberg_analysis.csv")
    
    # Plots
    params = {'N': N, 'R': R, 'L': L, 'B': B, 'rho': rho}
    
    plot_stackelberg_incentive(
        c_values, results['p_star'], results['total_incentive'],
        params=params, c_normalized=c_normalized,
        save=True, output_dir=FIGURES_DIR
    )
    
    plot_welfare_comparison(
        c_values, results['welfare_ne'], results['welfare_opt'], 
        results['welfare_stackelberg'],
        params=params, c_normalized=c_normalized,
        save=True, output_dir=FIGURES_DIR
    )
    
    # Check Stackelberg matches optimum
    matches = np.sum(results['k_matches_opt'])
    total = len(c_values)
    print(f"\nStackelberg achieves k_opt: {matches}/{total} cases ({100*matches/total:.1f}%)")
    
    return results


# =============================================================================
# EXPERIMENT 5: AoI vs k (with simulation validation)
# =============================================================================

def run_aoi_analysis():
    """Plot AoI vs k with analytical and simulated curves."""
    print_header("EXPERIMENT 5: AoI vs k")
    
    k_values = np.arange(1, 81)
    
    # Analytical
    aoi_analytical = aoi_vectorized(k_values, R, L)
    
    # Simulated (subset for speed)
    k_sim = list(range(1, 81, 5))  
    
    print(f"\nRunning Monte Carlo for k ∈ {{{k_sim[0]}, {k_sim[1]}, ..., {k_sim[-1]}}}...")
    
    sim_results = run_parameter_sweep(
        L=L, R=R, k_values=k_sim,
        T=5000, n_runs=200,
        show_progress=True,
        interior_target=True  
    )
    
    # Interpolate for plotting
    aoi_sim_full = np.interp(k_values, sim_results['k'], sim_results['mean_aoi'])
    aoi_std_full = np.interp(k_values, sim_results['k'], sim_results['mean_aoi_std'])
    
    # Plot
    params = {'R': R, 'L': L, 'rho': rho}
    plot_aoi_vs_k(
        k_values, aoi_analytical,
        aoi_simulated=aoi_sim_full,
        aoi_std=aoi_std_full,
        params=params,
        save=True, output_dir=FIGURES_DIR
    )
    
    # Save
    df = pd.DataFrame({
        'k': k_values,
        'aoi_analytical': aoi_analytical,
        'aoi_simulated': aoi_sim_full,
        'aoi_std': aoi_std_full,
    })
    df.to_csv(f'{RESULTS_DIR}/aoi_vs_k.csv', index=False)
    
    return sim_results


# =============================================================================
# EXPERIMENT 6: SPATIAL SNAPSHOT
# =============================================================================

def run_spatial_snapshot():
    """Generate spatial visualization."""
    print_header("EXPERIMENT 6: SPATIAL SNAPSHOT")
    
    from src.spatial import generate_positions, generate_target
    
    np.random.seed(42)
    
    # Generate positions
    positions = generate_positions(N, L, seed=42)
    target = generate_target(L, R, interior=True, seed=42)  # CORRECTION
    
    # Determine which would be active at equilibrium
    c_example = B_rho * 0.5  # Choose c where k* > 0
    k_star = find_nash_equilibrium(N, R, L, B, c_example)
    
    print(f"\nExample: c = {c_example:.4f}, k* = {k_star}")
    print(f"Target sampled from interior [{R}, {L-R}]²")
    
    # Randomly select k* volunteers to be "active"
    active_indices = np.random.choice(N, k_star, replace=False)
    active_mask = np.zeros(N, dtype=bool)
    active_mask[active_indices] = True
    
    # Plot
    plot_spatial_snapshot(
        positions, target, R, L,
        active_mask=active_mask,
        save=True, output_dir=FIGURES_DIR
    )
    
    return positions, target, active_mask


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all experiments."""
    print("\n" + "="*60)
    print("AoI CROWD-FINDING: EXPERIMENTAL SUITE (CORRECTED)")
    print("="*60)
    
    print(f"\nBase Parameters:")
    print(f"  N = {N} volunteers")
    print(f"  L = {L} m (area side)")
    print(f"  R = {R} m (detection radius)")
    print(f"  B = {B} (benefit)")
    print(f"  ρ = πR²/L² = {rho:.6f}")
    print(f"  Bρ = {B_rho:.4f} (NE threshold)")
    print(f"  NBρ = {NB_rho:.4f} (optimum threshold)")
    
    print("\n" + "-"*60)
    print("CORRECTIONS APPLIED:")
    print("  1. Target sampled from interior [R, L-R]²")
    print("  2. Cost ranges based on thresholds (Bρ, NBρ)")
    print("  3. PoA heatmap masks infinite values")
    print("  4. Stackelberg uses robust incentive calculation")
    print("-"*60)
    
    # Run experiments
    validation = run_validation()
    equilibrium = run_equilibrium_analysis()
    poa_results, poa_matrix = run_poa_analysis()
    stackelberg = run_stackelberg_analysis()
    aoi_results = run_aoi_analysis()
    spatial = run_spatial_snapshot()
    
    print_header("EXPERIMENTS COMPLETE")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")
    
    # Final summary
    print("\n" + "-"*60)
    print("SUMMARY:")
    print(f"  Validation: {np.sum(validation['within_tolerance'])}/{len(validation['k'])} passed")
    print(f"  Stackelberg matches optimal: {np.sum(stackelberg['k_matches_opt'])}/{len(stackelberg['c'])} cases")
    print("-"*60)


if __name__ == '__main__':
    main()
