"""Plotting utilities for experiments.

Contains plotting helpers used to generate publication-ready figures from experiment outputs.

Notes:
    PoA heatmaps mask or clip infinite values (typically when k* = 0) to keep plots interpretable.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
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

# Color scheme
COLORS = {
    'nash': '#1f77b4',      # Blue
    'optimum': '#2ca02c',   # Green
    'stackelberg': '#d62728', # Red
    'simulated': '#ff7f0e', # Orange
    'analytical': '#1f77b4', # Blue
    'inactive': '#7f7f7f',  # Gray
    'active': '#1f77b4',    # Blue
    'target': '#d62728',    # Red
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
    params: Dict = None,
    c_normalized: Optional[np.ndarray] = None,
    save: bool = True,
    output_dir: str = 'results/figures'
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
    c_normalized : array, optional
        Normalized cost values for x-axis (c/Bρ)
    save : bool
        Whether to save figure
    output_dir : str
        Output directory
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots()
    
    x = c_normalized if c_normalized is not None else c_values
    xlabel = '$c / (B\\rho)$' if c_normalized is not None else 'Cost $c$'
    
    ax.plot(x, k_ne, 'o-', color=COLORS['nash'], label='Nash Equilibrium $k^*$', markersize=3)
    ax.plot(x, k_opt, 's--', color=COLORS['optimum'], label='Social Optimum $k^{opt}$', markersize=3)
    
    # Mark threshold c = Bρ (normalized = 1)
    if c_normalized is not None:
        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='$c = B\\rho$')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Active Volunteers $k$')
    ax.legend(loc='best')
    
    if params:
        ax.set_title(f"N={params.get('N', '?')}, R/L={params.get('R', '?')/params.get('L', 1):.2f}")
    
    ax.set_ylim(bottom=0)
    
    if save:
        save_figure(fig, 'equilibrium_vs_cost', output_dir)
    
    return fig, ax


def plot_aoi_vs_k(
    k_values: np.ndarray,
    aoi_analytical: np.ndarray,
    aoi_simulated: Optional[np.ndarray] = None,
    aoi_std: Optional[np.ndarray] = None,
    params: Dict = None,
    save: bool = True,
    output_dir: str = 'results/figures'
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
        Standard error of simulated AoI
    params : dict
        Parameter info
    save : bool
        Whether to save
    output_dir : str
        Output directory
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots()
    
    # Filter out infinite values for plotting
    valid_mask = ~np.isinf(aoi_analytical)
    k_valid = k_values[valid_mask]
    aoi_valid = aoi_analytical[valid_mask]
    
    ax.plot(k_valid, aoi_valid, '-', color=COLORS['analytical'], 
            label='Analytical', linewidth=2)
    
    if aoi_simulated is not None:
        sim_valid = aoi_simulated[valid_mask]
        if aoi_std is not None:
            std_valid = aoi_std[valid_mask]
            ax.errorbar(k_valid, sim_valid, yerr=2*std_valid,
                       fmt='o', color=COLORS['simulated'], label='Simulated (±2σ)', 
                       capsize=3, markersize=4, alpha=0.8)
        else:
            ax.plot(k_valid, sim_valid, 'o', color=COLORS['simulated'], 
                   label='Simulated', markersize=4)
    
    ax.set_xlabel('Active Volunteers $k$')
    ax.set_ylabel('Expected AoI $\\bar{\\Delta}$')
    ax.legend(loc='best')
    
    # Use log scale if values span multiple orders of magnitude
    if len(aoi_valid) > 0 and aoi_valid.max() / aoi_valid.min() > 10:
        ax.set_yscale('log')
    
    if params:
        rho = params.get('rho', np.pi * params.get('R', 10)**2 / params.get('L', 100)**2)
        ax.set_title(f"$\\rho$ = {rho:.4f}")
    
    if save:
        save_figure(fig, 'aoi_vs_k', output_dir)
    
    return fig, ax


def plot_poa_heatmap(
    N_values: np.ndarray,
    c_values: np.ndarray,
    poa_matrix: np.ndarray,
    params: Dict = None,
    mask_infinite: bool = True,
    save: bool = True,
    output_dir: str = 'results/figures'
) -> Tuple[Figure, Axes]:
    """
    Plot Price of Anarchy heatmap.

    Notes:
        When mask_infinite=True, cells with infinite PoA (e.g., k*=0) are masked
        to keep the visualization interpretable.
    
    Parameters
    ----------
    N_values : array
        Population sizes (y-axis)
    c_values : array
        Cost values (x-axis)
    poa_matrix : 2D array
        PoA values (rows=N, cols=c)
    params : dict
        Parameter info
    mask_infinite : bool
        If True, mask cells where PoA is infinite (k*=0)
    save : bool
        Whether to save
    output_dir : str
        Output directory
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Create masked array for infinite values
    poa_plot = poa_matrix.copy()
    
    if mask_infinite:
        # Mask infinite and very large values
        poa_plot = np.ma.masked_where(
            np.isinf(poa_plot) | (poa_plot > 1e6), 
            poa_plot
        )
    
    # Check if we have any finite values
    if np.ma.count(poa_plot) == 0:
        ax.text(0.5, 0.5, 'All PoA values are infinite\n(k* = 0 everywhere)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_xlabel('Cost $c$')
        ax.set_ylabel('Population $N$')
        ax.set_title('Price of Anarchy (no finite values)')
    else:
        # Plot heatmap
        im = ax.imshow(
            poa_plot, 
            aspect='auto', 
            origin='lower',
            extent=[c_values[0], c_values[-1], N_values[0], N_values[-1]],
            cmap='YlOrRd',
            vmin=1.0,  # PoA >= 1 by definition
        )
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Price of Anarchy')
        
        # Add hatching for masked (infinite) regions
        if mask_infinite and np.ma.count_masked(poa_plot) > 0:
            # Create a second layer showing where PoA is infinite
            mask_overlay = np.ma.getmask(poa_plot).astype(float)
            ax.contourf(
                np.linspace(c_values[0], c_values[-1], poa_matrix.shape[1]),
                np.linspace(N_values[0], N_values[-1], poa_matrix.shape[0]),
                mask_overlay,
                levels=[0.5, 1.5],
                colors='none',
                hatches=['///'],
                alpha=0
            )
            # Add legend for hatched region
            hatched_patch = mpatches.Patch(facecolor='white', edgecolor='gray', 
                                           hatch='///', label='$k^* = 0$ (PoA = ∞)')
            ax.legend(handles=[hatched_patch], loc='upper right', fontsize=7)
        
        ax.set_xlabel('Cost $c$')
        ax.set_ylabel('Population $N$')
        ax.set_title('Price of Anarchy')
    
    if save:
        save_figure(fig, 'poa_heatmap', output_dir)
    
    return fig, ax


def plot_poa_vs_cost(
    c_values: np.ndarray,
    poa_values: np.ndarray,
    params: Dict = None,
    c_normalized: Optional[np.ndarray] = None,
    show_threshold: bool = True,
    save: bool = True,
    output_dir: str = 'results/figures'
) -> Tuple[Figure, Axes]:
    """
    Plot Price of Anarchy vs cost (line plot instead of heatmap).
    
    Parameters
    ----------
    c_values : array
        Cost values
    poa_values : array
        PoA values
    params : dict
        Parameter info
    c_normalized : array, optional
        Normalized cost values
    show_threshold : bool
        Show vertical line at c = Bρ
    save : bool
        Whether to save
    output_dir : str
        Output directory
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots()
    
    x = c_normalized if c_normalized is not None else c_values
    xlabel = '$c / (B\\rho)$' if c_normalized is not None else 'Cost $c$'
    
    # Filter finite values
    finite_mask = np.isfinite(poa_values) & (poa_values < 1e6)
    
    if np.any(finite_mask):
        ax.plot(x[finite_mask], poa_values[finite_mask], 'o-', 
               color=COLORS['nash'], markersize=4)
    
    # Mark infinite region
    if np.any(~finite_mask):
        ax.axvspan(x[~finite_mask].min(), x[~finite_mask].max(), 
                  alpha=0.2, color='gray', label='$k^* = 0$ (PoA = ∞)')
    
    if show_threshold and c_normalized is not None:
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, 
                  label='$c = B\\rho$ (threshold)')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Price of Anarchy')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(loc='best')
    
    if params:
        ax.set_title(f"N={params.get('N', '?')}")
    
    if save:
        save_figure(fig, 'poa_vs_cost', output_dir)
    
    return fig, ax


def plot_stackelberg_incentive(
    c_values: np.ndarray,
    p_star: np.ndarray,
    total_incentive: np.ndarray,
    params: Dict = None,
    c_normalized: Optional[np.ndarray] = None,
    save: bool = True,
    output_dir: str = 'results/figures'
) -> Tuple[Figure, Tuple[Axes, Axes]]:
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
    c_normalized : array, optional
        Normalized cost values
    save : bool
        Whether to save
    output_dir : str
        Output directory
    
    Returns
    -------
    fig, (ax1, ax2) : Figure and Axes
    """
    apply_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    
    x = c_normalized if c_normalized is not None else c_values
    xlabel = '$c / (B\\rho)$' if c_normalized is not None else 'Cost $c$'
    
    # Per-volunteer incentive
    ax1.plot(x, p_star, 'o-', color=COLORS['stackelberg'], markersize=3)
    ax1.plot(x, c_values, 'k--', alpha=0.5, label='$p = c$ (full subsidy)')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Optimal Incentive $p^*$')
    ax1.legend(loc='best', fontsize=7)
    ax1.set_title('Per-Volunteer Incentive')
    
    # Total incentive
    ax2.plot(x, total_incentive, 's-', color=COLORS['stackelberg'], markersize=3)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Total Payment $p^* \\cdot k^{opt}$')
    ax2.set_title('Platform Total Cost')
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'stackelberg_incentive', output_dir)
    
    return fig, (ax1, ax2)


def plot_welfare_comparison(
    c_values: np.ndarray,
    W_ne: np.ndarray,
    W_opt: np.ndarray,
    W_stack: np.ndarray,
    params: Dict = None,
    c_normalized: Optional[np.ndarray] = None,
    save: bool = True,
    output_dir: str = 'results/figures'
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
    c_normalized : array, optional
        Normalized cost values
    save : bool
        Whether to save
    output_dir : str
        Output directory
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots()
    
    x = c_normalized if c_normalized is not None else c_values
    xlabel = '$c / (B\\rho)$' if c_normalized is not None else 'Cost $c$'
    
    ax.plot(x, W_ne, 'o-', color=COLORS['nash'], label='Nash Equilibrium', markersize=3)
    ax.plot(x, W_opt, '--', color=COLORS['optimum'], label='Social Optimum', linewidth=2)
    ax.plot(x, W_stack, 's-', color=COLORS['stackelberg'], label='Stackelberg', 
           markersize=3, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Social Welfare $W$')
    ax.legend(loc='best')
    ax.set_title('Welfare Comparison')
    
    if save:
        save_figure(fig, 'welfare_comparison', output_dir)
    
    return fig, ax


def plot_participation_gap(
    c_values: np.ndarray,
    k_ne: np.ndarray,
    k_opt: np.ndarray,
    params: Dict = None,
    c_normalized: Optional[np.ndarray] = None,
    save: bool = True,
    output_dir: str = 'results/figures'
) -> Tuple[Figure, Axes]:
    """
    Plot participation gap (k_opt - k*) vs cost.
    
    Parameters
    ----------
    c_values : array
        Cost values
    k_ne : array
        Nash equilibrium participation
    k_opt : array
        Social optimum participation
    params : dict
        Parameter info
    c_normalized : array, optional
        Normalized cost values
    save : bool
        Whether to save
    output_dir : str
        Output directory
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots()
    
    x = c_normalized if c_normalized is not None else c_values
    xlabel = '$c / (B\\rho)$' if c_normalized is not None else 'Cost $c$'
    
    gap = k_opt - k_ne
    
    ax.plot(x, gap, 'o-', color='purple', markersize=4)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Participation Gap $k^{opt} - k^*$')
    ax.set_title('Under-participation at Nash Equilibrium')
    
    # Annotate theoretical gap
    if params:
        N = params.get('N', 100)
        rho = params.get('rho', 0.0314)
        theoretical_gap = np.log(N) / rho if rho > 0 else 0
        ax.axhline(y=theoretical_gap, color='red', linestyle='--', alpha=0.5,
                  label=f'Theory: $\\ln(N)/\\rho \\approx {theoretical_gap:.1f}$')
        ax.legend()
    
    if save:
        save_figure(fig, 'participation_gap', output_dir)
    
    return fig, ax


def plot_spatial_snapshot(
    positions: np.ndarray,
    target: np.ndarray,
    R: float,
    L: float,
    active_mask: Optional[np.ndarray] = None,
    save: bool = True,
    output_dir: str = 'results/figures'
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
    output_dir : str
        Output directory
    
    Returns
    -------
    fig, ax : Figure and Axes
    """
    apply_style()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Interior region boundary (where target is sampled)
    interior_rect = plt.Rectangle((R, R), L - 2*R, L - 2*R, 
                                   fill=False, color='green', 
                                   linestyle='--', linewidth=1,
                                   label='Target sampling region')
    ax.add_patch(interior_rect)
    
    # Target and detection zone
    circle = plt.Circle(target, R, fill=False, color=COLORS['target'], 
                        linestyle='-', linewidth=2, label='Detection Zone')
    ax.add_patch(circle)
    ax.plot(target[0], target[1], '*', color=COLORS['target'], 
           markersize=15, label='Target')
    
    # Volunteers
    if active_mask is None:
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c=COLORS['active'], s=20, alpha=0.6, label='Volunteers')
    else:
        n_active = np.sum(active_mask)
        n_inactive = np.sum(~active_mask)
        if n_inactive > 0:
            ax.scatter(positions[~active_mask, 0], positions[~active_mask, 1],
                      c=COLORS['inactive'], s=15, alpha=0.4, 
                      label=f'Inactive ({n_inactive})')
        if n_active > 0:
            ax.scatter(positions[active_mask, 0], positions[active_mask, 1],
                      c=COLORS['active'], s=25, alpha=0.8, 
                      label=f'Active ({n_active})')
    
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_aspect('equal')
    ax.set_title('Spatial Configuration')
    
    if save:
        save_figure(fig, 'spatial_snapshot', output_dir)
    
    return fig, ax


def plot_validation_results(
    k_values: np.ndarray,
    analytical: np.ndarray,
    simulated: np.ndarray,
    simulated_std: np.ndarray,
    passed: np.ndarray,
    save: bool = True,
    output_dir: str = 'results/figures'
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Plot validation results: analytical vs simulated AoI.
    
    Parameters
    ----------
    k_values : array
        Number of active volunteers
    analytical : array
        Analytical AoI
    simulated : array
        Simulated AoI
    simulated_std : array
        Standard error of simulation
    passed : array of bool
        Whether each k passed validation
    save : bool
        Whether to save
    output_dir : str
        Output directory
    
    Returns
    -------
    fig, (ax1, ax2) : Figure and Axes
    """
    apply_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    
    # Left: AoI comparison
    valid = ~np.isinf(analytical)
    ax1.plot(k_values[valid], analytical[valid], '-', color=COLORS['analytical'],
            label='Analytical', linewidth=2)
    ax1.errorbar(k_values[valid], simulated[valid], yerr=2*simulated_std[valid],
                fmt='o', color=COLORS['simulated'], label='Simulated (±2σ)',
                capsize=3, markersize=4)
    ax1.set_xlabel('Active Volunteers $k$')
    ax1.set_ylabel('Expected AoI $\\bar{\\Delta}$')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title('AoI: Theory vs Simulation')
    
    # Right: Relative error
    rel_error = np.abs(simulated - analytical) / analytical
    rel_error[~valid] = 0
    
    colors = ['green' if p else 'red' for p in passed[valid]]
    ax2.bar(k_values[valid], rel_error[valid] * 100, color=colors, alpha=0.7)
    ax2.axhline(y=5, color='red', linestyle='--', label='5% threshold')
    ax2.set_xlabel('Active Volunteers $k$')
    ax2.set_ylabel('Relative Error (%)')
    ax2.legend()
    ax2.set_title(f'Validation: {np.sum(passed)}/{len(passed)} passed')
    
    plt.tight_layout()
    
    if save:
        save_figure(fig, 'validation_results', output_dir)
    
    return fig, (ax1, ax2)
