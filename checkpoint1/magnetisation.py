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
def get_delta_e(spins, x, y, L, J):
    """Energy change for flipping spin at (x, y) with periodic boundaries."""
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
def calculate_magnetization(spins):
    """Calculates total magnetization of the lattice."""
    return np.sum(spins)

# ------- Main Simulation Routine -------

def run_simulation():
    L = 50                  
    N = L * L               
    J = 1.0                 
    T_range = np.arange(1.0, 3.1, 0.1)  
    n_equil = 500           # We can afford more sweeps now!
    n_measurements = 1000   
    n_decorrelate = 10      

    avg_mag_list = []
    suscep_list = []

    # Sequential heating: start ordered at T=1.0
    spins = np.ones((L, L), dtype=np.int32) 

    print(f"Starting Numba-accelerated simulation on {L}x{L} lattice...")
    print(f"{'T':<5} | {'<|m|>':<10} | {'chi':<10}")
    print("-" * 30)

    for T in T_range:
        beta = 1.0 / T
        m_values = np.zeros(n_measurements)

        # 1. Equilibration
        for _ in range(n_equil):
            mc_sweep(spins, L, N, beta, J)

        # 2. Measurement
        for i in range(n_measurements):
            for _ in range(n_decorrelate):
                mc_sweep(spins, L, N, beta, J)
            m_values[i] = calculate_magnetization(spins)

        # 3. Analysis
        avg_abs_m = np.mean(np.abs(m_values)) / N
        mag_variance = np.mean(m_values**2) - np.mean(np.abs(m_values))**2
        chi = (1.0 / (N * T)) * mag_variance

        avg_mag_list.append(avg_abs_m)
        suscep_list.append(chi)
        print(f"{T:5.1f} | {avg_abs_m:10.4f} | {chi:10.4f}")

    return avg_mag_list, suscep_list

# Run the simulation
magnetization, susceptibility = run_simulation()
T_range = np.arange(1.0, 3.1, 0.1)

# ------- Saving Data -------
data_path = os.path.join(data_dir, "task4_glauber_data.txt")
header = "Temperature(kBT)  Avg_Abs_Magnetization  Susceptibility"
np.savetxt(data_path, np.column_stack((T_range, magnetization, susceptibility)), header=header)
print(f"\nData saved to: {data_path}")

# ------- Plotting & Saving Figures -------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot 1: Magnetization
ax1.plot(T_range, magnetization, 'o-', color='tab:blue', label='Simulated')
ax1.axvline(x=2.269, color='black', linestyle='--', label=r'Theory $T_c \approx 2.27$')
ax1.set_xlabel(r'$k_B T$')
ax1.set_ylabel(r'$\langle |M| \rangle / N$')
ax1.set_title('Magnetisation vs Temperature (Glauber)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Susceptibility
ax2.plot(T_range, susceptibility, 's-', color='tab:red', label='Simulated')
ax2.axvline(x=2.269, color='black', linestyle='--', label=r'Theory $T_c \approx 2.27$')
ax2.set_xlabel(r'$k_B T$')
ax2.set_ylabel(r'Susceptibility $\chi$')
ax2.set_title('Susceptibility vs Temperature')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
figure_path = os.path.join(figures_dir, "task4_glauber_plots.png")
plt.savefig(figure_path)
plt.show()