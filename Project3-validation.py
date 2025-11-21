#!/usr/bin/env python3
"""
Standalone script for Ising Model Validation and Advanced Diagnostics
Parts 8-9: Theoretical validation and comprehensive MCMC diagnostics

This script can run independently after you've implemented the core Ising model.
Just make sure the functions from parts 1-7 are available to import.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Try to import the core Ising model functions
try:
    # If your main code is in a file called ising_model.py, import from there
    # Otherwise, adjust the import based on your file structure
    from Project3 import (
        IsingModel2D,
        metropolis_step,
        metropolis_simulation,
        wolff_simulation,
        autocorrelation_time,
        calculate_observables,
        exact_magnetization_onsager,
        gelman_rubin_diagnostic
    )
    print("✓ Successfully imported core Ising model functions")
except ImportError as e:
    print("✗ Failed to import core functions. Make sure your main Ising code is available.")
    print(f"  Error: {e}")
    print("\nIf your code is in a different file, edit the import statement at the top.")
    sys.exit(1)

# ============================================================================
# Part 8: Theoretical Predictions and Validation
# ============================================================================

def theoretical_predictions():
    """
    Return exact theoretical predictions for 2D Ising model at Tc.
    Based on Onsager's exact solution and scaling theory.
    """
    Tc = 2.269185
    
    predictions = {
        'Tc': Tc,
        'magnetization': 0.0,  # m = 0 at Tc (continuous phase transition)
        'energy': -1.4527004,  # Exact: -2/π * (1 + √2)
        'specific_heat': float('inf'),  # Logarithmic divergence
        'susceptibility': float('inf'),  # Power-law divergence
        
        # Critical exponents (exact)
        'beta': 1/8,      # m ~ (Tc - T)^β for T < Tc
        'gamma': 7/4,     # χ ~ |T - Tc|^(-γ)
        'alpha': 0.0,     # C ~ log|T - Tc| (α = 0 means log divergence)
        'nu': 1.0,        # Correlation length ξ ~ |T - Tc|^(-ν)
        
        'finite_size_note': 'Finite lattices show rounded transition with finite peaks'
    }
    
    return predictions

def validate_against_theory(L, n_steps=6250, equilibration=3125):
    """
    Validate simulation results against exact theoretical predictions.
    """
    Tc = 2.269185
    theory = theoretical_predictions()
    
    print("\n" + "="*70)
    print("VALIDATION AGAINST THEORETICAL PREDICTIONS")
    print("="*70)
    
    print(f"\nRunning simulations at Tc = {Tc:.6f}")
    print(f"Lattice size: {L}×{L}")
    print(f"Steps: {n_steps} (after {equilibration} equilibration steps)")
    print("\n" + "-"*70)
    
    # Run simulations
    print("\nRunning Metropolis algorithm...")
    metro_data = metropolis_simulation(L, Tc, n_steps, equilibration)
    metro_obs = calculate_observables(metro_data['magnetization'], 
                                     metro_data['energy'], Tc, L)
    
    print("Running Wolff algorithm...")
    wolff_data = wolff_simulation(L, Tc, n_steps//5, equilibration//5)
    wolff_obs = calculate_observables(wolff_data['magnetization'],
                                     wolff_data['energy'], Tc, L)
    
    # Create comparison table
    print("\n" + "="*70)
    print("RESULTS COMPARISON AT CRITICAL TEMPERATURE")
    print("="*70)
    
    print("\n1. MAGNETIZATION")
    print("-"*70)
    print(f"{'Quantity':<25} {'Value':<15} {'Notes':<30}")
    print("-"*70)
    print(f"{'Theoretical (exact)':<25} {theory['magnetization']:<15.6f} {'m = 0 at Tc':<30}")
    print(f"{'Metropolis':<25} {metro_obs['magnetization']:.6f} ± {metro_obs['magnetization_error']:.6f}")
    print(f"{'Wolff':<25} {wolff_obs['magnetization']:.6f} ± {wolff_obs['magnetization_error']:.6f}")
    print(f"{'Finite-size effect':<25} {'~L^(-β/ν)':<15} {'= L^(-1/8) for 2D Ising':<30}")
    
    # Expected finite-size magnetization
    m_finite_size = L**(-1/8)
    print(f"{'Expected for L={L}':<25} {m_finite_size:<15.6f} {'(finite-size scaling)':<30}")
    
    print("\n2. ENERGY PER SPIN")
    print("-"*70)
    print(f"{'Quantity':<25} {'Value':<15} {'Notes':<30}")
    print("-"*70)
    print(f"{'Theoretical (exact)':<25} {theory['energy']:<15.6f}")
    e_exact_detailed = -2/np.pi * (1 + np.sqrt(2))
    print(f"{'Exact formula':<25} {e_exact_detailed:<15.6f} {'-2/π(1+√2)':<30}")
    print(f"{'Metropolis':<25} {metro_obs['energy']:.6f} ± {metro_obs['energy_error']:.6f}")
    print(f"{'Wolff':<25} {wolff_obs['energy']:.6f} ± {wolff_obs['energy_error']:.6f}")
    
    # Calculate relative errors
    metro_e_error = abs(metro_obs['energy'] - theory['energy']) / abs(theory['energy']) * 100
    wolff_e_error = abs(wolff_obs['energy'] - theory['energy']) / abs(theory['energy']) * 100
    print(f"{'Metropolis rel. error':<25} {metro_e_error:<15.3f} {'%':<30}")
    print(f"{'Wolff rel. error':<25} {wolff_e_error:<15.3f} {'%':<30}")
    
    print("\n3. SPECIFIC HEAT")
    print("-"*70)
    print(f"{'Quantity':<25} {'Value':<15} {'Notes':<30}")
    print("-"*70)
    print(f"{'Theoretical':<25} {'diverges':<15} {'C ~ log|T-Tc| (α=0)':<30}")
    print(f"{'Metropolis':<25} {metro_obs['specific_heat']:<15.3f} {'(rounded by finite size)':<30}")
    print(f"{'Wolff':<25} {wolff_obs['specific_heat']:<15.3f}")
    print(f"{'Finite-size scaling':<25} {'C_max ~ log(L)':<15} {'Peak height grows with L':<30}")
    
    print("\n4. MAGNETIC SUSCEPTIBILITY")
    print("-"*70)
    print(f"{'Quantity':<25} {'Value':<15} {'Notes':<30}")
    print("-"*70)
    print(f"{'Theoretical':<25} {'diverges':<15} {'χ ~ |T-Tc|^(-7/4)':<30}")
    print(f"{'Metropolis':<25} {metro_obs['susceptibility']:<15.3f} {'(rounded by finite size)':<30}")
    print(f"{'Wolff':<25} {wolff_obs['susceptibility']:<15.3f}")
    
    print("\n5. AUTOCORRELATION TIME")
    print("-"*70)
    print(f"{'Algorithm':<25} {'τ (magnetization)':<20} {'τ (energy)':<20}")
    print("-"*70)
    print(f"{'Metropolis':<25} {metro_obs['tau_magnetization']:<20.2f} {metro_obs['tau_energy']:<20.2f}")
    print(f"{'Wolff':<25} {wolff_obs['tau_magnetization']:<20.2f} {wolff_obs['tau_energy']:<20.2f}")
    speedup = metro_obs['tau_magnetization'] / wolff_obs['tau_magnetization']
    print(f"{'Wolff speedup':<25} {speedup:<20.2f}x")
    
    print("\n6. CRITICAL EXPONENTS (Exact values)")
    print("-"*70)
    print(f"{'Exponent':<15} {'Value':<15} {'Physical meaning':<40}")
    print("-"*70)
    print(f"{'β':<15} {theory['beta']:<15} {'m ~ (Tc-T)^β':<40}")
    print(f"{'γ':<15} {theory['gamma']:<15} {'χ ~ |T-Tc|^(-γ)':<40}")
    print(f"{'α':<15} {theory['alpha']:<15} {'C ~ log|T-Tc| (logarithmic)':<40}")
    print(f"{'ν':<15} {theory['nu']:<15} {'ξ ~ |T-Tc|^(-ν)':<40}")
    
    print("\n" + "="*70)
    
    return {
        'theory': theory,
        'metropolis': metro_obs,
        'wolff': wolff_obs,
        'metro_data': metro_data,
        'wolff_data': wolff_data
    }

# ============================================================================
# Part 9: Advanced Diagnostics and Visualization
# ============================================================================

def plot_trace_diagnostics(data_dict, algorithm_name, T):
    """
    Create comprehensive trace plots for MCMC diagnostics.
    Shows raw traces and running means.
    """
    mag = data_dict['magnetization']
    energy = data_dict['energy']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Calculate running means
    def running_mean(x, window=50):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window
    
    steps = np.arange(len(mag))
    
    # Magnetization trace
    ax = axes[0, 0]
    ax.plot(steps, mag, 'b-', alpha=0.5, linewidth=0.5)
    ax.set_xlabel('MC Step')
    ax.set_ylabel('|m|')
    ax.set_title(f'{algorithm_name}: Magnetization Trace')
    ax.grid(True, alpha=0.3)
    
    # Magnetization running mean
    ax = axes[0, 1]
    window = min(50, len(mag)//10)
    if len(mag) > window:
        running_mag = running_mean(mag, window)
        ax.plot(steps[window-1:], running_mag, 'b-', linewidth=1.5)
        ax.axhline(np.mean(mag), color='r', linestyle='--', label=f'Mean = {np.mean(mag):.4f}')
    ax.set_xlabel('MC Step')
    ax.set_ylabel('|m| (running mean)')
    ax.set_title(f'Running Mean (window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy trace
    ax = axes[1, 0]
    ax.plot(steps, energy, 'g-', alpha=0.5, linewidth=0.5)
    ax.set_xlabel('MC Step')
    ax.set_ylabel('E/N')
    ax.set_title(f'{algorithm_name}: Energy Trace')
    ax.grid(True, alpha=0.3)
    
    # Energy running mean
    ax = axes[1, 1]
    if len(energy) > window:
        running_e = running_mean(energy, window)
        ax.plot(steps[window-1:], running_e, 'g-', linewidth=1.5)
        ax.axhline(np.mean(energy), color='r', linestyle='--', label=f'Mean = {np.mean(energy):.4f}')
    ax.set_xlabel('MC Step')
    ax.set_ylabel('E/N (running mean)')
    ax.set_title(f'Running Mean (window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{algorithm_name} Trace Plots at T={T:.3f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_multiple_chains(chains, observable_name, T, algorithm='Metropolis'):
    """
    Plot multiple independent chains to visualize convergence.
    """
    n_chains = len(chains)
    
    fig, axes = plt.subplots(n_chains + 1, 1, figsize=(12, 2.5*n_chains + 2))
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_chains))
    
    # Individual chain plots
    for i, chain in enumerate(chains):
        ax = axes[i]
        steps = np.arange(len(chain))
        ax.plot(steps, chain, color=colors[i], alpha=0.7, linewidth=0.8)
        ax.axhline(np.mean(chain), color=colors[i], linestyle='--', linewidth=2, 
                  label=f'Mean = {np.mean(chain):.4f}')
        ax.set_ylabel(f'Chain {i+1}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        if i < n_chains - 1:
            ax.set_xticklabels([])
    
    axes[-2].set_xlabel('MC Step')
    
    # Combined plot
    ax = axes[-1]
    for i, chain in enumerate(chains):
        steps = np.arange(len(chain))
        ax.plot(steps, chain, color=colors[i], alpha=0.5, linewidth=0.8, label=f'Chain {i+1}')
    
    # Overall mean
    all_data = np.concatenate(chains)
    overall_mean = np.mean(all_data)
    ax.axhline(overall_mean, color='black', linestyle='--', linewidth=2, 
              label=f'Overall mean = {overall_mean:.4f}')
    
    ax.set_xlabel('MC Step')
    ax.set_ylabel('All Chains')
    ax.legend(loc='upper right', ncol=min(n_chains+1, 5))
    ax.grid(True, alpha=0.3)
    
    # Calculate R-hat
    R_hat = gelman_rubin_diagnostic(chains)
    status = 'CONVERGED ✓' if R_hat < 1.1 else 'NOT CONVERGED ✗'
    
    plt.suptitle(f'{algorithm} - Multiple Chains for {observable_name} at T={T:.3f}\n' + 
                f'Gelman-Rubin R-hat = {R_hat:.4f} ({status})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return R_hat

def plot_configuration_evolution(L, T, n_snapshots=6, n_steps_between=250):
    """
    Show how spin configuration evolves over time.
    """
    model = IsingModel2D(L)
    beta = 1.0 / T
    N = L * L
    
    # Determine grid layout
    n_cols = 3
    n_rows = (n_snapshots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_snapshots > 1 else [axes]
    
    # Initial configuration
    axes[0].imshow(model.spins, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title(f'Step 0 (Initial)')
    axes[0].axis('off')
    
    # Evolution
    for idx in range(1, n_snapshots):
        # Run some steps
        for _ in range(n_steps_between):
            for _ in range(N):
                metropolis_step(model, beta)
        
        axes[idx].imshow(model.spins, cmap='RdBu', vmin=-1, vmax=1)
        axes[idx].set_title(f'Step {idx * n_steps_between}')
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(n_snapshots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Configuration Evolution at T={T:.3f} ({L}×{L} lattice)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def check_convergence(L, T, n_chains=4, n_steps=2500):
    """
    Run multiple chains to check convergence.
    """
    print(f"\nConvergence check at T={T:.3f} with {n_chains} chains")
    print("-"*40)
    
    chains = []
    for i in range(n_chains):
        print(f"Running chain {i+1}/{n_chains}...")
        data = metropolis_simulation(L, T, n_steps, n_steps//2)
        chains.append(data['magnetization'])
    
    R_hat = gelman_rubin_diagnostic(chains)
    
    print(f"\nGelman-Rubin R-hat = {R_hat:.4f}")
    print(f"Status: {'CONVERGED' if R_hat < 1.1 else 'NOT CONVERGED'} (threshold: R-hat < 1.1)")
    
    return chains, R_hat

def comprehensive_diagnostics(L=32, T=None, n_steps=3750, n_chains=4):
    """
    Run comprehensive MCMC diagnostics including traces, multiple chains, and configurations.
    """
    if T is None:
        T = 2.269185  # Critical temperature
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MCMC DIAGNOSTICS")
    print("="*70)
    print(f"\nParameters: L={L}, T={T:.4f}, n_steps={n_steps}, n_chains={n_chains}")
    
    # 1. Single chain trace plots
    print("\n1. Running single chain for trace diagnostics...")
    metro_data = metropolis_simulation(L, T, n_steps, n_steps//2)
    plot_trace_diagnostics(metro_data, 'Metropolis', T)
    
    # 2. Multiple chains for convergence
    print("\n2. Running multiple chains for convergence check...")
    print(f"   Running {n_chains} independent chains...")
    
    mag_chains = []
    energy_chains = []
    
    for i in range(n_chains):
        print(f"   Chain {i+1}/{n_chains}...")
        data = metropolis_simulation(L, T, n_steps, n_steps//2)
        mag_chains.append(data['magnetization'])
        energy_chains.append(data['energy'])
    
    print("\n   Plotting magnetization chains...")
    R_hat_mag = plot_multiple_chains(mag_chains, 'Magnetization', T)
    
    print("\n   Plotting energy chains...")
    R_hat_energy = plot_multiple_chains(energy_chains, 'Energy', T)
    
    print(f"\n   Convergence Results:")
    print(f"   - Magnetization R-hat = {R_hat_mag:.4f} {'✓' if R_hat_mag < 1.1 else '✗'}")
    print(f"   - Energy R-hat = {R_hat_energy:.4f} {'✓' if R_hat_energy < 1.1 else '✗'}")
    
    # 3. Configuration evolution
    print("\n3. Generating configuration evolution snapshots...")
    plot_configuration_evolution(L, T, n_snapshots=6, n_steps_between=250)
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main function to run validation and diagnostics.
    """
    print("="*70)
    print("ISING MODEL - THEORETICAL VALIDATION & DIAGNOSTICS")
    print("="*70)
    
    # Parameters
    L = 32
    Tc = 2.269185
    
    print("\nWhat would you like to run?")
    print("1. Theoretical validation only (Part 8)")
    print("2. MCMC diagnostics only (Part 9)")
    print("3. Both validation and diagnostics (recommended)")
    print("4. Quick test (validation + basic diagnostics)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Just validation
        print("\nRunning theoretical validation...")
        validation_results = validate_against_theory(L, n_steps=6250, equilibration=3125)
        
    elif choice == '2':
        # Just diagnostics
        print("\nRunning comprehensive diagnostics...")
        comprehensive_diagnostics(L, Tc, n_steps=3750, n_chains=4)
        
    elif choice == '3':
        # Full analysis
        print("\n" + "="*70)
        print("PART 8: THEORETICAL VALIDATION")
        print("="*70)
        validation_results = validate_against_theory(L, n_steps=6250, equilibration=3125)
        
        print("\n" + "="*70)
        print("PART 9: COMPREHENSIVE DIAGNOSTICS")
        print("="*70)
        
        # Trace plots
        print("\nGenerating trace plots...")
        plot_trace_diagnostics(validation_results['metro_data'], 'Metropolis', Tc)
        plot_trace_diagnostics(validation_results['wolff_data'], 'Wolff', Tc)
        
        # Multiple chains
        print("\nChecking convergence with multiple chains...")
        chains, R_hat = check_convergence(L, Tc, n_chains=4, n_steps=2500)
        plot_multiple_chains(chains, 'Magnetization', Tc)
        
        # Configuration evolution
        print("\nGenerating configuration evolution at different temperatures...")
        plot_configuration_evolution(L, T=1.5, n_snapshots=6, n_steps_between=187)
        plot_configuration_evolution(L, T=Tc, n_snapshots=6, n_steps_between=187)
        plot_configuration_evolution(L, T=3.0, n_snapshots=6, n_steps_between=187)
        
    elif choice == '4':
        # Quick test with fewer steps
        print("\nRunning quick validation (fewer steps)...")
        validation_results = validate_against_theory(L, n_steps=2000, equilibration=1000)
        
        print("\nGenerating basic diagnostics...")
        plot_trace_diagnostics(validation_results['metro_data'], 'Metropolis', Tc)
        
        chains, R_hat = check_convergence(L, Tc, n_chains=3, n_steps=1500)
        plot_multiple_chains(chains, 'Magnetization', Tc)
        
    else:
        print("Invalid choice. Running quick test by default...")
        validation_results = validate_against_theory(L, n_steps=2000, equilibration=1000)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()