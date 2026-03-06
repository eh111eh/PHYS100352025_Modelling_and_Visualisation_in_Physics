import numpy as np
import matplotlib.pyplot as plt
import os

# Create directories for files
os.makedirs('data', exist_ok=True)
os.makedirs('fig', exist_ok=True)

def update_gol(lattice):
    """Deterministic parallel update with Periodic Boundary Conditions."""
    neighbors = (
        np.roll(np.roll(lattice, 1, 0), 1, 1) + np.roll(np.roll(lattice, 1, 0), 0, 1) + 
        np.roll(np.roll(lattice, 1, 0), -1, 1) + np.roll(np.roll(lattice, 0, 0), 1, 1) + 
        np.roll(np.roll(lattice, 0, 0), -1, 1) + np.roll(np.roll(lattice, -1, 0), 1, 1) + 
        np.roll(np.roll(lattice, -1, 0), 0, 1) + np.roll(np.roll(lattice, -1, 0), -1, 1)
    )
    new_lattice = np.zeros_like(lattice)
    new_lattice[(lattice == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
    new_lattice[(lattice == 0) & (neighbors == 3)] = 1
    return new_lattice

# --- Simulation Setup ---
N = 50
lattice = np.zeros((N, N))
# Glider pattern: moves 1 step right/down every 4 iterations 
glider = np.array([[0,1,0], [0,0,1], [1,1,1]])
lattice[5:8, 5:8] = glider

times, com_positions = [], []

# Simulation loop
prev_com = None
unwrapped_com = np.array([5.0, 5.0])

for t in range(200):
    coords = np.argwhere(lattice == 1)
    if len(coords) > 0:
        # --- 1. CoM calculation with PBC ---
        theta = coords * (2 * np.pi / N)
        cos_mean = np.mean(np.cos(theta), axis=0)
        sin_mean = np.mean(np.sin(theta), axis=0)
        
        # Obtain angle using atan2 and convert to the coordinate (0~N)
        current_com = np.arctan2(sin_mean, cos_mean) * (N / (2 * np.pi))
        current_com = np.mod(current_com, N)
        
        # --- 2. Unwrapping ---
        if prev_com is not None:
            diff = current_com - prev_com
            
            # PBC Correction: Minimum Image Convention
            for i in range(2):
                if diff[i] > N/2: # prev = 0.8, curr = 49.7 -> diff = +48.9, corrected diff = -1.1 to the left
                    diff[i] -= N
                elif diff[i] < -N/2: # prev = 49.2, curr = 0.3 -> diff = -48.9, corrected diff = +1.1 to the right
                    diff[i] += N
            
            unwrapped_com = unwrapped_com + diff # corrected diff is added
        
        times.append(t)
        com_positions.append(unwrapped_com.copy())
        prev_com = current_com.copy()
        
    lattice = update_gol(lattice)

times = np.array(times)
com_positions = np.array(com_positions)

# 1. Net Speed (Long-term)
displacements = np.sqrt(np.sum((com_positions - com_positions[0])**2, axis=1))
fit_coeffs = np.polyfit(times, displacements, 1)
net_speed_fit = fit_coeffs[0]

# 2. Instantaneous Velocities
inst_speeds = np.sqrt(np.sum(np.diff(com_positions, axis=0)**2, axis=1))
mean_inst_velocity = np.mean(inst_speeds)

# --- Save Data Files ---
np.savetxt("data/glider_displacement.csv", np.column_stack((times, displacements)), 
           header="Time,Net_Displacement", delimiter=",")
np.savetxt("data/glider_velocity.csv", np.column_stack((times[1:], inst_speeds)), 
           header="Time,Inst_Velocity", delimiter=",")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Displacement vs Time (Net Speed)
ax1.plot(times, displacements, 'o', markersize=4, label='CoM Displacement')
ax1.plot(times, np.polyval(fit_coeffs, times), 'r-', 
         label=f'Net Speed Fit (v ≈ {net_speed_fit:.4f})')
ax1.set_title("Glider True Net Speed Over The Long Term")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Displacement (cells)")
ax1.legend()
ax1.grid(True, linestyle='--')

# Plot 2: Velocity vs Time (True Instantaneous)
ax2.plot(times[1:], inst_speeds, 'g-o', markersize=3, alpha=0.7, label='Inst. Velocity')
# ax2.axhline(y=net_speed_fit, color='r', linestyle='--', label=f'Net Speed ({net_speed_fit:.4f})')
ax2.axhline(y=mean_inst_velocity, color='blue', linestyle='-.', label=f'Mean Inst. Speed ({mean_inst_velocity:.4f})')
ax2.set_title("Glider Instantaneous Speed")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Velocity (cells/step)")
ax2.set_ylim(0, 1.0) 
ax2.legend()
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("fig/glider_true_velocity_analysis.png")
plt.show()

print(f"Net Speed (Fit): {net_speed_fit:.4f}")
print(f"Mean Instantaneous Speed: {mean_inst_velocity:.4f}")