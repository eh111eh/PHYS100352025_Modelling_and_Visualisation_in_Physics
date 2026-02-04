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
            # Right and Down neighbors to avoid double counting
            nb = spins[(x + 1) % L, y] + spins[x, (y + 1) % L]
            energy += -J * s * nb
    return energy

@njit
def get_delta_e(spins, x, y, L, J):
    """Energy change for flipping spin at (x, y)."""
    s = spins[x, y]
    nb = (spins[(x + 1) % L, y] + spins[(x - 1) % L, y] + 
          spins[x, (y + 1) % L] + spins[x, (y - 1) % L])
    return 2.0 * J * s * nb

@njit
def mc_sweep(spins, L, N, beta, J):
    """Performs one full Monte Carlo sweep (N update attempts)."""
    for _ in range(N):
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        dE = get_delta_e(spins, x, y, L, J)
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[x, y] *= -1

@njit
def bootstrap_heat_capacity(e_values, T, N, n_resamples=200):
    """Numba-accelerated Bootstrap for Specific Heat error."""
    n_data = len(e_values)
    c_resamples = np.zeros(n_resamples)
    
    for i in range(n_resamples):
        # Sample with replacement
        resample_sum_e = 0.0
        resample_sum_e2 = 0.0
        for _ in range(n_data):
            val = e_values[np.random.randint(0, n_data)]
            resample_sum_e += val
            resample_sum_e2 += val**2
            
        mean_e = resample_sum_e / n_data
        mean_e2 = resample_sum_e2 / n_data
        var_e = mean_e2 - mean_e**2
        c_resamples[i] = var_e / (N * T**2)
    
    return np.mean(c_resamples), np.std(c_resamples)

# ------- Main Simulation Routine -------

def run_simulation():
    L = 50
    N = L * L
    J = 1.0
    T_range = np.arange(1.0, 3.1, 0.1)
    n_equil = 500         # Increased since it's now very fast
    n_measurements = 2000 # Increased for better statistics
    n_decorrelate = 10
    
    energy_list = []
    energy_err_list = []
    heat_cap_list = []
    heat_cap_err_list = []

    spins = np.ones((L, L), dtype=np.int32) 

    print(f"Starting Numba-accelerated simulation on {L}x{L} lattice...")
    print(f"{'T':<5} | {'<E>/N':<10} | {'C':<10}")
    print("-" * 35)

    for T in T_range:
        beta = 1.0 / T
        e_values = np.zeros(n_measurements)

        # 1. Equilibration
        for _ in range(n_equil):
            mc_sweep(spins, L, N, beta, J)

        # 2. Measurement
        for i in range(n_measurements):
            for _ in range(n_decorrelate):
                mc_sweep(spins, L, N, beta, J)
            e_values[i] = get_total_energy(spins, L, J)

        # 3. Analysis
        avg_e = np.mean(e_values) / N
        e_err = np.std(e_values) / (N * np.sqrt(n_measurements))
        
        c_val, c_err = bootstrap_heat_capacity(e_values, T, N)

        energy_list.append(avg_e)
        energy_err_list.append(e_err)
        heat_cap_list.append(c_val)
        heat_cap_err_list.append(c_err)
        
        print(f"{T:5.1f} | {avg_e:10.4f} | {c_val:10.4f}")

    return T_range, energy_list, energy_err_list, heat_cap_list, heat_cap_err_list

# Execute and Save
T_range, energy, energy_err, heat_cap, heat_cap_err = run_simulation()

# Saving Data
data_path = os.path.join(data_dir, "task5_numba_data.txt")
np.savetxt(data_path, np.column_stack((T_range, energy, energy_err, heat_cap, heat_cap_err)), 
           header="T  Avg_Energy/N  Energy_Err  Heat_Capacity  HC_Err")

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.errorbar(T_range, energy, yerr=energy_err, fmt='o-', color='navy', capsize=3)
ax1.set_ylabel(r'$\langle E \rangle / N$')
ax1.set_title('Internal Energy vs Temperature')
ax1.grid(True, alpha=0.3)

ax2.errorbar(T_range, heat_cap, yerr=heat_cap_err, fmt='s-', color='crimson', capsize=3)
ax2.axvline(x=2.269, color='black', linestyle='--', label='Theory $T_c$')
ax2.set_xlabel(r'$k_B T$')
ax2.set_ylabel(r'Heat Capacity $C$')
ax2.set_title('Specific Heat with Bootstrap Error Bars')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "task5_numba_plots.png"))
plt.show()