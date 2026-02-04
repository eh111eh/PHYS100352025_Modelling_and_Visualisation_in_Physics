import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit

# ------- Directory Setup -------
base_dir = "checkpoint1"
data_dir = os.path.join(base_dir, "data")
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# ------- Numba Accelerated Kernels -------

@njit
def get_total_energy(spins, L, J):
    """Calculates total energy with periodic boundaries."""
    energy = 0.0
    for x in range(L):
        for y in range(L):
            s = spins[x, y]
            # Sum right and down to avoid double counting
            nb = spins[(x + 1) % L, y] + spins[x, (y + 1) % L]
            energy += -J * s * nb
    return energy

@njit
def get_neighbours_sum(spins, x, y, L):
    """Sum of 4 nearest neighbours."""
    return (spins[(x + 1) % L, y] + spins[(x - 1) % L, y] + 
            spins[x, (y + 1) % L] + spins[x, (y - 1) % L])

@njit
def kawasaki_sweep(spins, L, beta, J):
    """Performs one full Monte Carlo sweep using Kawasaki exchange."""
    N = L * L
    for _ in range(N):
        # Pick two random sites
        x1, y1 = np.random.randint(0, L), np.random.randint(0, L)
        x2, y2 = np.random.randint(0, L), np.random.randint(0, L)
        
        s1, s2 = spins[x1, y1], spins[x2, y2]
        
        if s1 == s2:
            continue
            
        # Check if they are nearest neighbours
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        is_nn = (dx == 1 and dy == 0) or (dx == 0 and dy == 1) or \
                (dx == L-1 and dy == 0) or (dx == 0 and dy == L-1)
        
        if is_nn:
            # Correction: subtracting the mutual bond to avoid double counting
            dE = 2 * J * (s1 * (get_neighbours_sum(spins, x1, y1, L) - s2) + 
                          s2 * (get_neighbours_sum(spins, x2, y2, L) - s1))
        else:
            dE = 2 * J * (s1 * get_neighbours_sum(spins, x1, y1, L) + 
                          s2 * get_neighbours_sum(spins, x2, y2, L))
            
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[x1, y1], spins[x2, y2] = s2, s1

@njit
def bootstrap_heat_capacity(e_values, T, N, n_resamples=200):
    """Calculates specific heat error using Bootstrap."""
    n_data = len(e_values)
    c_resamples = np.zeros(n_resamples)
    for i in range(n_resamples):
        sample_sum = 0.0
        sample_sq_sum = 0.0
        for _ in range(n_data):
            val = e_values[np.random.randint(0, n_data)]
            sample_sum += val
            sample_sq_sum += val**2
        var_e = (sample_sq_sum / n_data) - (sample_sum / n_data)**2
        c_resamples[i] = var_e / (N * T**2)
    return np.mean(c_resamples), np.std(c_resamples)

# ------- Main Simulation -------

def run_kawasaki_task6():
    L = 50
    N = L * L
    J = 1.0
    T_range = np.arange(1.0, 3.6, 0.2)
    n_equil = 1000        # Kawasaki takes longer to reach equilibrium
    n_measurements = 1000
    n_decorrelate = 20    # Higher decorrelation for exchange dynamics
    
    # Initialize random lattice (M stays near 0)
    spins = np.random.choice(np.array([-1, 1], dtype=np.int32), size=(L, L))
    
    results = []

    print(f"Running Kawasaki simulation ({L}x{L})...")
    print(f"{'T':<5} | {'<E>/N':<10} | {'C':<10}")
    
    for T in T_range:
        beta = 1.0 / T
        e_values = np.zeros(n_measurements)
        
        # 1. Equilibration
        for _ in range(n_equil):
            kawasaki_sweep(spins, L, beta, J)
            
        # 2. Measurement
        for i in range(n_measurements):
            for _ in range(n_decorrelate):
                kawasaki_sweep(spins, L, beta, J)
            e_values[i] = get_total_energy(spins, L, J)
            
        avg_e = np.mean(e_values) / N
        e_err = np.std(e_values) / (N * np.sqrt(n_measurements))
        c_val, c_err = bootstrap_heat_capacity(e_values, T, N)
        
        results.append([T, avg_e, e_err, c_val, c_err])
        print(f"{T:5.1f} | {avg_e:10.4f} | {c_val:10.4f}")

    return np.array(results)

# Execute
data = run_kawasaki_task6()
T_axis = data[:, 0]

# Save Data
np.savetxt(os.path.join(data_dir, "task6_kawasaki_data.txt"), data, 
           header="T Avg_E/N E_err Heat_Capacity C_err")

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Energy
ax1.errorbar(T_axis, data[:, 1], yerr=data[:, 2], fmt='o-', color='purple')
ax1.set_ylabel(r'$\langle E \rangle / N$')
ax1.set_title('Average Energy vs Temperature (Kawasaki)')
ax1.grid(True, alpha=0.3)

# Heat Capacity
ax2.errorbar(T_axis, data[:, 3], yerr=data[:, 4], fmt='s-', color='orange')
ax2.axvline(x=2.269, color='black', linestyle='--', label='Theory $T_c$')
ax2.set_xlabel(r'$k_B T$')
ax2.set_ylabel(r'Heat Capacity $C$')
ax2.set_title('Specific Heat with Bootstrap Errors')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "task6_kawasaki_plots.png"))
plt.show()