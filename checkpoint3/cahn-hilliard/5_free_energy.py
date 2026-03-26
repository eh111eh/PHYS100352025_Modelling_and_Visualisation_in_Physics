"""
Task 5: Free Energy vs Time
============================
Runs the Cahn-Hilliard simulation for two initial conditions
(phi0 = 0 and phi0 = 0.5) and plots the free energy over time.

The free energy density is defined as:
    f = -phi^2/2 + phi^4/4 + |grad phi|^2 / 2

We expect F to decrease monotonically over time as the system
evolves towards phase separation (energy minimisation).

Output: checkpoint3/cahn-hilliard/fig/task5_free_energy.png
        checkpoint3/cahn-hilliard/data/task5_free_energy.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# ── Save path ─────────────────────────────────────────────────────────────────

SAVE_DIR = "checkpoint3/cahn-hilliard"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────

N             = 100      # Grid size (N x N)
dx            = 1.0      # Dimensionless spatial step
dt            = 0.01     # Dimensionless time step (stable for dx=1)
noise_amp     = 0.01     # Initial noise amplitude
n_steps       = 50000    # Total simulation steps
record_every  = 200      # Record F every this many steps

# ── Core functions ────────────────────────────────────────────────────────────

def laplacian(f):
    return (
        np.roll(f,  1, axis=0) + np.roll(f, -1, axis=0) +
        np.roll(f,  1, axis=1) + np.roll(f, -1, axis=1) -
        4 * f
    ) / dx**2

def step(phi):
    mu = -phi * (1 - phi**2) - laplacian(phi)
    return phi + dt * laplacian(mu)

def free_energy(phi):
    """
    Total dimensionless free energy:
        F = integral[ -phi^2/2 + phi^4/4 + |grad phi|^2/2 ] dx^2
    Gradient computed via centred differences.
    """
    gx = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
    gy = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    f  = -0.5 * phi**2 + 0.25 * phi**4 + 0.5 * (gx**2 + gy**2)
    return np.sum(f) * dx**2

# ── Run simulation for a given phi0, collect F(step) ─────────────────────────

def run(phi0, label):
    np.random.seed(42)
    phi = phi0 + noise_amp * np.random.randn(N, N)

    steps    = []
    energies = []

    print(f"Running phi0 = {phi0} ...")
    for n in range(1, n_steps + 1):
        phi = step(phi)
        if n % record_every == 0:
            steps.append(n)                      # record step number, not time
            energies.append(free_energy(phi))
            if n % (record_every * 25) == 0:
                print(f"  step {n:6d} | F = {energies[-1]:.4f}")

    print(f"  Done. Final F = {energies[-1]:.4f}\n")
    return np.array(steps), np.array(energies)

# ── Run both cases ────────────────────────────────────────────────────────────

steps_0,   F_0   = run(phi0=0.0,  label="phi0=0")
steps_05,  F_05  = run(phi0=0.5,  label="phi0=0.5")

# ── Save CSV ──────────────────────────────────────────────────────────────────

csv_path = os.path.join(SAVE_DIR, "data/task5_free_energy.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "F_phi0_0", "F_phi0_05"])
    for s, f0, f05 in zip(steps_0, F_0, F_05):
        writer.writerow([s, f"{f0:.6f}", f"{f05:.6f}"])
print(f"CSV saved: {csv_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0f0f1a')

for ax, steps, F, phi0_label, color in zip(
    axes,
    [steps_0,   steps_05],
    [F_0,       F_05],
    ['φ₀ = 0',  'φ₀ = 0.5'],
    ['#4cc9f0', '#f4a261']
):
    ax.set_facecolor('#0f0f1a')
    ax.plot(steps, F, color=color, linewidth=1.5, label=phi0_label)

    # Mark the equilibrium value (mean of last 10% of steps)
    F_eq = np.mean(F[int(len(F)*0.9):])
    ax.axhline(F_eq, color='white', linewidth=1, linestyle='--', alpha=0.6,
               label=f'Equilibrium  F ≈ {F_eq:.2f}')

    ax.set_xlabel('Step', color='white', fontsize=11)        # x axis = step
    ax.set_ylabel('Free Energy  F', color='white', fontsize=11)
    ax.set_title(f'Free Energy vs Step  ({phi0_label})', color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

plt.suptitle('Cahn-Hilliard: Free Energy Minimisation (Task 5)',
             color='white', fontsize=14, y=1.01)
plt.tight_layout()

plot_path = os.path.join(SAVE_DIR, "fig/task5_free_energy.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
print(f"Plot saved: {plot_path}")
plt.show()