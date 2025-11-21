#!/usr/bin/env python3
"""
2D Ising Model with MCMC Sampling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

# Part 1: The Ising Model
class IsingModel2D:
    """
    2D Ising model on a square lattice.
    Spins can be +1 (up) or -1 (down).
    Uses periodic boundary conditions.
    """
    
    def __init__(self, L, J=1.0, h=0.0):
        """
        Parameters:
        -----------
        L : int
            Linear size of the lattice (L x L grid)
        J : float
            Coupling constant (interaction strength)
        h : float
            External magnetic field
        """
        self.L = L
        self.N = L * L  # Total number of spins
        self.J = J
        self.h = h
        
        # Initialize with random spins
        self.spins = 2 * np.random.randint(2, size=(L, L)) - 1
        
    def energy(self):
        """Calculate total energy of the system."""
        # Get neighbor sums efficiently using roll
        neighbor_sum = (np.roll(self.spins, 1, axis=0) + 
                       np.roll(self.spins, -1, axis=0) +
                       np.roll(self.spins, 1, axis=1) + 
                       np.roll(self.spins, -1, axis=1))
        
        # Energy = -J * sum of neighbor interactions - h * sum of spins
        return -self.J * np.sum(self.spins * neighbor_sum) / 2 - self.h * np.sum(self.spins)
    
    def magnetization(self):
        """Calculate total magnetization."""
        return np.sum(self.spins)
    
    def energy_diff(self, i, j):
        """
        Calculate energy difference if we flip spin at (i,j).
        This is much faster than calculating full energy twice.
        """
        # Get neighbors with periodic boundary conditions
        top = self.spins[(i-1) % self.L, j]
        bottom = self.spins[(i+1) % self.L, j]
        left = self.spins[i, (j-1) % self.L]
        right = self.spins[i, (j+1) % self.L]
        
        neighbors = top + bottom + left + right
        
        # Energy difference = 2 * spin * (J * neighbors + h)
        return 2 * self.spins[i, j] * (self.J * neighbors + self.h)

# Part 2: Metropolis Algorithm
def metropolis_step(model, beta):
    """
    Perform one Metropolis Monte Carlo step.
    
    Parameters:
    -----------
    model : IsingModel2D
        The Ising model
    beta : float
        Inverse temperature (1/T)
    
    Returns:
    --------
    bool : Whether the flip was accepted
    """
    # Choose random spin
    i = np.random.randint(model.L)
    j = np.random.randint(model.L)
    
    # Calculate energy change
    dE = model.energy_diff(i, j)
    
    # Metropolis acceptance rule
    if dE <= 0:
        model.spins[i, j] *= -1
        return True
    elif np.random.random() < np.exp(-beta * dE):
        model.spins[i, j] *= -1
        return True
    
    return False

def metropolis_simulation(L, T, n_steps, equilibration_steps=None):
    """
    Run Metropolis algorithm simulation.
    
    Parameters:
    -----------
    L : int
        Lattice size
    T : float
        Temperature
    n_steps : int
        Number of MC sweeps (1 sweep = N spin flip attempts)
    equilibration_steps : int
        Steps to equilibrate before measuring (default: n_steps // 2)
    
    Returns:
    --------
    dict : Dictionary with magnetization and energy time series
    """
    if equilibration_steps is None:
        equilibration_steps = n_steps // 2
    
    model = IsingModel2D(L)
    beta = 1.0 / T
    N = L * L
    
    # Storage for measurements
    magnetization = []
    energy = []
    
    # Equilibration (no measurements)
    for step in range(equilibration_steps):
        for _ in range(N):  # One sweep
            metropolis_step(model, beta)
    
    # Production (with measurements)
    for step in range(n_steps):
        # One sweep
        for _ in range(N):
            metropolis_step(model, beta)
        
        # Measure every sweep
        m = model.magnetization() / N
        e = model.energy() / N
        magnetization.append(abs(m))  # Use absolute magnetization
        energy.append(e)
    
    return {
        'magnetization': np.array(magnetization),
        'energy': np.array(energy),
        'final_config': model.spins.copy()
    }


# Part 3: Wolff Cluster Algorithm
def wolff_cluster_flip(model, beta):
    """
    Perform one Wolff cluster flip.
    
    This algorithm flips entire clusters of aligned spins,
    dramatically reducing critical slowing down.
    
    Returns:
    --------
    int : Size of the flipped cluster
    """
    L = model.L
    J = model.J
    
    # Probability to add aligned neighbor to cluster
    p_add = 1 - np.exp(-2 * beta * J)
    
    # Choose random seed spin
    i_seed = np.random.randint(L)
    j_seed = np.random.randint(L)
    
    # Initialize cluster
    cluster = set()
    stack = [(i_seed, j_seed)]
    old_spin = model.spins[i_seed, j_seed]
    
    # Build cluster using depth-first search
    while stack:
        i, j = stack.pop()
        
        if (i, j) in cluster:
            continue
            
        cluster.add((i, j))
        
        # Check all 4 neighbors
        neighbors = [
            ((i-1) % L, j),
            ((i+1) % L, j),
            (i, (j-1) % L),
            (i, (j+1) % L)
        ]
        
        for ni, nj in neighbors:
            if (ni, nj) not in cluster:
                if model.spins[ni, nj] == old_spin:
                    if np.random.random() < p_add:
                        stack.append((ni, nj))
    
    # Flip all spins in cluster
    for i, j in cluster:
        model.spins[i, j] *= -1
    
    return len(cluster)

def wolff_simulation(L, T, n_steps, equilibration_steps=None):
    """
    Run Wolff cluster algorithm simulation.
    
    Parameters:
    -----------
    L : int
        Lattice size
    T : float
        Temperature
    n_steps : int
        Number of cluster flips
    equilibration_steps : int
        Steps to equilibrate before measuring
    
    Returns:
    --------
    dict : Dictionary with measurements and cluster sizes
    """
    if equilibration_steps is None:
        equilibration_steps = n_steps // 2
    
    model = IsingModel2D(L)
    beta = 1.0 / T
    N = L * L
    
    magnetization = []
    energy = []
    cluster_sizes = []
    
    # Equilibration
    for _ in range(equilibration_steps):
        wolff_cluster_flip(model, beta)
    
    # Production
    for step in range(n_steps):
        cluster_size = wolff_cluster_flip(model, beta)
        cluster_sizes.append(cluster_size)
        
        # Measure
        m = model.magnetization() / N
        e = model.energy() / N
        magnetization.append(abs(m))
        energy.append(e)
    
    return {
        'magnetization': np.array(magnetization),
        'energy': np.array(energy),
        'cluster_sizes': np.array(cluster_sizes),
        'final_config': model.spins.copy()
    }


# Part 4: Analysis Tools

def autocorrelation_time(data, max_lag=None):
    """
    Calculate integrated autocorrelation time.
    
    This tells us how many MC steps correspond to one independent sample.
    """
    if max_lag is None:
        max_lag = min(len(data) // 4, 250)
    
    data = np.array(data)
    mean = np.mean(data)
    var = np.var(data)
    
    if var == 0:
        return 1.0
    
    # Calculate autocorrelation function
    tau = 0.0
    for k in range(1, max_lag):
        # Autocorrelation at lag k
        acf = np.mean((data[:-k] - mean) * (data[k:] - mean)) / var
        
        if acf < 0.05:  # Stop when correlation becomes negligible
            break
            
        tau += acf
    
    return 1 + 2 * tau




def calculate_observables(magnetization, energy, T, L):
    """
    Calculate thermodynamic observables with error estimates.
    """
    beta = 1.0 / T
    N = L * L
    
    # Mean values
    m_mean = np.mean(magnetization)
    m2_mean = np.mean(magnetization**2)
    e_mean = np.mean(energy)
    e2_mean = np.mean(energy**2)
    
    # Susceptibility and specific heat
    chi = beta * N * (m2_mean - m_mean**2)
    C = beta**2 * N * (e2_mean - e_mean**2)
    
    # Error estimates
    m_error = np.std(magnetization) / np.sqrt(len(magnetization))
    e_error = np.std(energy) / np.sqrt(len(energy))
    
    return {
        'magnetization': m_mean,
        'magnetization_error': m_error,
        'energy': e_mean,
        'energy_error': e_error,
        'susceptibility': chi,
        'specific_heat': C
    }

def exact_magnetization_onsager(T, Tc=2.269185):
    """
    Onsager's exact solution for magnetization in 2D Ising model.
    """
    if T >= Tc:
        return 0.0
    else:
        beta = 1.0 / T
        return (1 - np.sinh(2*beta)**(-4))**(1/8)

# Part 5: Temperature Scan
def temperature_scan(L=32, T_min=1.5, T_max=3.5, n_temps=15):
    """
    Scan across temperatures to observe phase transition.
    """
    Tc = 2.269185  # Critical temperature
    
    # Create evenly spaced temperature array
    temps = np.linspace(T_min, T_max, n_temps)
    
    
    # Finding closest temperature to Tc and replacing it with exact Tc
    idx_closest = np.argmin(np.abs(temps - Tc))
    temps[idx_closest] = Tc
    
    results_metro = []
    results_wolff = []
    
    print(f"Temperature scan for L={L} lattice")
    print("="*60)
    
    for T in temps:
        print(f"\nT = {T:.3f}:")
        
        # Determine number of steps based on proximity to Tc
        if abs(T - Tc) < 0.1:
            n_metro = 3750
            n_wolff = 625
            equil_metro = 1875
            equil_wolff = 312
        elif abs(T - Tc) < 0.3:
            n_metro = 2500
            n_wolff = 375
            equil_metro = 1250
            equil_wolff = 187
        else:
            n_metro = 1250
            n_wolff = 250
            equil_metro = 625
            equil_wolff = 125
        
        # Metropolis
        print(f"  Running Metropolis ({n_metro} sweeps)...")
        start = time.time()
        metro_data = metropolis_simulation(L, T, n_metro, equil_metro)
        metro_time = time.time() - start
        metro_obs = calculate_observables(
            metro_data['magnetization'], 
            metro_data['energy'], T, L
        )
        metro_obs['time'] = metro_time
        results_metro.append(metro_obs)
        
        # Wolff
        print(f"  Running Wolff ({n_wolff} steps)...")
        start = time.time()
        wolff_data = wolff_simulation(L, T, n_wolff, equil_wolff)
        wolff_time = time.time() - start
        wolff_obs = calculate_observables(
            wolff_data['magnetization'],
            wolff_data['energy'], T, L
        )
        wolff_obs['time'] = wolff_time
        results_wolff.append(wolff_obs)
        
        print(f"  Metropolis: |m| = {metro_obs['magnetization']:.3f}")
        print(f"  Wolff:      |m| = {wolff_obs['magnetization']:.3f}")
    
    return temps, results_metro, results_wolff

if __name__ == "__main__":
    L = 32  # for simulation
    temps, results_metro, results_wolff = temperature_scan(L=L)

    # For plotting
    m_metro = [r['magnetization'] for r in results_metro]
    m_wolff = [r['magnetization'] for r in results_wolff]

    e_metro = [r['energy'] for r in results_metro]
    e_wolff = [r['energy'] for r in results_wolff]

    C_metro = [r['specific_heat'] for r in results_metro]
    C_wolff = [r['specific_heat'] for r in results_wolff]

    chi_metro = [r['susceptibility'] for r in results_metro]
    chi_wolff = [r['susceptibility'] for r in results_wolff]

    # Plots
    plt.figure(figsize=(14, 10))

    # Magnetization
    plt.subplot(2, 2, 1)
    plt.plot(temps, m_metro, 'o-', label="Metropolis")
    plt.plot(temps, m_wolff, 's-', label="Wolff")
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization |m|")
    plt.title("Magnetization vs Temperature")
    plt.legend()

    # Energy
    plt.subplot(2, 2, 2)
    plt.plot(temps, e_metro, 'o-', label="Metropolis")
    plt.plot(temps, e_wolff, 's-', label="Wolff")
    plt.xlabel("Temperature T")
    plt.ylabel("Energy per spin")
    plt.title("Energy vs Temperature")
    plt.legend()

    # Specific Heat
    plt.subplot(2, 2, 3)
    plt.plot(temps, C_metro, 'o-', label='Metropolis')
    plt.plot(temps, C_wolff, 's-', label='Wolff')
    plt.xlabel("Temperature T")
    plt.ylabel("Specific Heat C")
    plt.title("Specific Heat vs Temperature")
    plt.legend()

    # Susceptibility
    plt.subplot(2, 2, 4)
    plt.plot(temps, chi_metro, 'o-', label='Metropolis')
    plt.plot(temps, chi_wolff, 's-', label='Wolff')
    plt.xlabel("Temperature T")
    plt.ylabel("Susceptibility χ")
    plt.title("Susceptibility vs Temperature")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================================
# Part 6: Convergence Diagnostics
# ============================================================================

def gelman_rubin_diagnostic(chains):
    """
    Calculate Gelman-Rubin R-hat statistic for convergence.
    
    Parameters:
    -----------
    chains : list of arrays
        Multiple independent MCMC chains
    
    Returns:
    --------
    float : R-hat value (should be < 1.1 for convergence)
    """
    m = len(chains)  # Number of chains
    n = len(chains[0])  # Length of each chain
    
    # Calculate within-chain variance
    W = np.mean([np.var(chain, ddof=1) for chain in chains])
    
    # Calculate between-chain variance
    chain_means = [np.mean(chain) for chain in chains]
    B = n * np.var(chain_means, ddof=1)
    
    # Calculate R-hat
    var_plus = ((n-1)/n) * W + (1/n) * B
    R_hat = np.sqrt(var_plus / W) if W > 0 else 1.0
    
    return R_hat

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

# ============================================================================
# Part 7: Visualization
# ============================================================================

def plot_phase_transition(temps, results_metro, results_wolff):
    """
    Create comprehensive plots showing the phase transition.
    """
    Tc = 2.269185
    
    # Extract data
    m_metro = [r['magnetization'] for r in results_metro]
    m_wolff = [r['magnetization'] for r in results_wolff]
    chi_metro = [r['susceptibility'] for r in results_metro]
    C_metro = [r['specific_heat'] for r in results_metro]
    tau_metro = [r['tau_magnetization'] for r in results_metro]
    tau_wolff = [r['tau_magnetization'] for r in results_wolff]
    
    # Exact solution
    T_exact = np.linspace(1.5, Tc, 100)
    m_exact = [exact_magnetization_onsager(T) for T in T_exact]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Magnetization
    ax = axes[0, 0]
    ax.plot(temps, m_metro, 'bo-', label='Metropolis', markersize=4)
    ax.plot(temps, m_wolff, 'rs-', label='Wolff', markersize=4, alpha=0.7)
    ax.plot(T_exact, m_exact, 'g--', label='Onsager exact', linewidth=2)
    ax.axvline(Tc, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('|m|')
    ax.set_title('Magnetization per spin')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy
    ax = axes[0, 1]
    e_metro = [r['energy'] for r in results_metro]
    ax.plot(temps, e_metro, 'go-', markersize=4)
    ax.axvline(Tc, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('E/N')
    ax.set_title('Energy per spin')
    ax.grid(True, alpha=0.3)
    
    # Susceptibility
    ax = axes[0, 2]
    ax.plot(temps, chi_metro, 'mo-', markersize=4)
    ax.axvline(Tc, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('χ')
    ax.set_title('Magnetic Susceptibility')
    ax.grid(True, alpha=0.3)
    
    # Specific heat
    ax = axes[1, 0]
    ax.plot(temps, C_metro, 'co-', markersize=4)
    ax.axvline(Tc, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('C')
    ax.set_title('Specific Heat')
    ax.grid(True, alpha=0.3)
    
    # Autocorrelation time
    ax = axes[1, 1]
    ax.semilogy(temps, tau_metro, 'bo-', label='Metropolis', markersize=4)
    ax.semilogy(temps, tau_wolff, 'rs-', label='Wolff', markersize=4)
    ax.axvline(Tc, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('τ (log scale)')
    ax.set_title('Autocorrelation Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Speedup factor
    ax = axes[1, 2]
    speedup = np.array(tau_metro) / np.array(tau_wolff)
    ax.plot(temps, speedup, 'ko-', markersize=4)
    ax.axvline(Tc, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Speedup')
    ax.set_title('Wolff Speedup (τ_Metro / τ_Wolff)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('2D Ising Model Phase Transition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_configurations(L, temps_to_plot=[1.5, 2.269, 3.0]):
    """
    Show spin configurations at different temperatures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, T in enumerate(temps_to_plot):
        # Run simulation - INCREASED BY 25%
        if T == 2.269:
            n_steps = 2500
        else:
            n_steps = 1250
            
        data = metropolis_simulation(L, T, n_steps, n_steps//2)
        config = data['final_config']
        
        # Plot
        ax = axes[idx]
        im = ax.imshow(config, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f'T = {T:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        if idx == 0:
            ax.text(0.5, -0.15, 'Ordered\n(T < Tc)', 
                   transform=ax.transAxes, ha='center')
        elif idx == 1:
            ax.text(0.5, -0.15, 'Critical\n(T = Tc)', 
                   transform=ax.transAxes, ha='center')
        else:
            ax.text(0.5, -0.15, 'Disordered\n(T > Tc)', 
                   transform=ax.transAxes, ha='center')
    
    plt.suptitle(f'Spin Configurations ({L}×{L} lattice)', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Spin', ticks=[-1, 0, 1])
    plt.tight_layout()
    plt.show()
#Hani
# ============================================================================
# Part 8: Theoretical Predictions
# ============================================================================

def theoretical_predictions():
    """
    Return exact theoretical predictions for 2D Ising model at Tc.
    Based on Onsager's exact solution and scaling theory.
    """
    Tc = 2.269185
    
    # Exact values at Tc
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
        
        # For finite size systems
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
    Run complete analysis of 2D Ising model.
    """
    print("="*60)
    print("2D ISING MODEL - MCMC ANALYSIS (FAST MODE)")
    print("="*60)
    
    # Parameters
    L = 32  # Lattice size
    Tc = 2.269185
    
    # 1. Quick test at critical temperature
    print("\n1. Quick comparison at critical temperature")
    print("-"*60)
    
    print(f"Running at Tc = {Tc:.4f} on {L}×{L} lattice\n")
    
    # Metropolis - INCREASED BY 25%
    print("Metropolis (2500 sweeps)...")
    metro_data = metropolis_simulation(L, Tc, 2500, 1250)
    metro_obs = calculate_observables(metro_data['magnetization'], 
                                     metro_data['energy'], Tc, L)
    
    # Wolff - INCREASED BY 25%
    print("Wolff (625 steps)...")
    wolff_data = wolff_simulation(L, Tc, 625, 312)
    wolff_obs = calculate_observables(wolff_data['magnetization'],
                                     wolff_data['energy'], Tc, L)
    
    print("\nResults at Tc:")
    print(f"{'Algorithm':<12} {'|m|':<8} {'τ':<8} {'N_eff':<8}")
    print("-"*36)
    print(f"{'Metropolis':<12} {metro_obs['magnetization']:<8.3f} "
          f"{metro_obs['tau_magnetization']:<8.1f} "
          f"{metro_obs['n_eff']:<8.1f}")
    print(f"{'Wolff':<12} {wolff_obs['magnetization']:<8.3f} "
          f"{wolff_obs['tau_magnetization']:<8.1f} "
          f"{wolff_obs['n_eff']:<8.1f}")
    print(f"{'Onsager':<12} {exact_magnetization_onsager(Tc):<8.3f}")
    
    # 2. Validation against theory
    print("\n2. Validation against theoretical predictions")
    print("-"*60)
    validation_results = validate_against_theory(L, n_steps=6250, equilibration=3125)
    
    # 3. Trace plots and diagnostics
    print("\n3. MCMC Trace Diagnostics")
    print("-"*60)
    print("Generating trace plots for Metropolis...")
    plot_trace_diagnostics(validation_results['metro_data'], 'Metropolis', Tc)
    
    print("Generating trace plots for Wolff...")
    plot_trace_diagnostics(validation_results['wolff_data'], 'Wolff', Tc)
    
    # 4. Multiple chains convergence check
    print("\n4. Multiple Chain Convergence Analysis")
    print("-"*60)
    chains, R_hat = check_convergence(L, Tc, n_chains=4, n_steps=2500)
    plot_multiple_chains(chains, 'Magnetization', Tc)
    
    # 5. Configuration evolution - INCREASED BY 25%
    print("\n5. Configuration Evolution at Different Temperatures")
    print("-"*60)
    print("Showing evolution at T < Tc (ordered)...")
    plot_configuration_evolution(L, T=1.5, n_snapshots=6, n_steps_between=187)
    
    print("Showing evolution at T = Tc (critical)...")
    plot_configuration_evolution(L, T=Tc, n_snapshots=6, n_steps_between=187)
    
    print("Showing evolution at T > Tc (disordered)...")
    plot_configuration_evolution(L, T=3.0, n_snapshots=6, n_steps_between=187)
    
    # 6. Temperature scan
    print("\n6. Temperature scan")
    print("-"*60)
    temps, results_metro, results_wolff = temperature_scan(L, n_temps=15)
    
    # 7. Phase transition plots
    print("\n7. Creating phase transition plots...")
    plot_phase_transition(temps, results_metro, results_wolff)
    
    # 8. Final configurations at different temperatures
    print("\n8. Final configurations at different temperatures...")
    plot_configurations(L)
    
    # 9. Optional: Comprehensive diagnostics at Tc
    print("\n9. Run comprehensive diagnostics at Tc? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        comprehensive_diagnostics(L, Tc, n_steps=3750, n_chains=4)
    
    print("\n"+"="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()