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