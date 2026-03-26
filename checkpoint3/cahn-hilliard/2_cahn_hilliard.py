"""
Cahn-Hilliard Equation Simulator (Task 2)
==========================================
Solves the dimensionless Cahn-Hilliard equation numerically
and visualises the result as a real-time animation.

Dimensionless equations (Task 1 result):
    d(phi)/dt = nabla^2(mu)
    mu        = -phi*(1 - phi^2) - nabla^2(phi)

Algorithm: Explicit Euler + 2D centred difference (periodic boundary conditions)
Grid:      100 x 100
"""

import numpy as np
import matplotlib.pyplot as plt

# ── 1. Parameters ─────────────────────────────────────────────────────────────

N   = 100       # Grid size (N x N)
dx  = 1.0       # Dimensionless spatial step
dt  = 0.01      # Dimensionless time step
                # * Stability: Cahn-Hilliard has a nabla^2(nabla^2(phi)) structure,
                #   so 1/dx^2 is applied twice. The formal condition is
                #   dt <= dx^4 / (8*(dx^2+4)) = 0.025 for dx=1, but the nonlinear
                #   term (-phi^3) tightens this in practice, so dt=0.01 is safe.

phi0      = -0.5    # Initial mean composition (0: fully mixed / +-0.5: biased)
noise_amp = 0.01   # Initial noise amplitude (smaller = more physical)

n_steps       = 50000   # Total number of simulation steps
plot_interval = 100     # Update the plot every this many steps

# ── 2. Initial condition ──────────────────────────────────────────────────────

np.random.seed(42)
phi = phi0 + noise_amp * np.random.randn(N, N)

# ── 3. Core functions ─────────────────────────────────────────────────────────

def laplacian(f):
    """
    2D Laplacian via centred differences with periodic boundary conditions.
    np.roll handles the periodicity automatically.
    """
    return (
        np.roll(f,  1, axis=0) +
        np.roll(f, -1, axis=0) +
        np.roll(f,  1, axis=1) +
        np.roll(f, -1, axis=1) -
        4 * f
    ) / dx**2


def compute_mu(phi):
    """Dimensionless chemical potential: mu = -phi*(1 - phi^2) - nabla^2(phi)"""
    return -phi * (1 - phi**2) - laplacian(phi)


def step(phi):
    """One time step via Explicit Euler: phi += dt * nabla^2(mu)"""
    return phi + dt * laplacian(compute_mu(phi))


def free_energy(phi):
    """
    Dimensionless free energy (for Task 5)
    F = sum[ -phi^2/2 + phi^4/4 + |grad phi|^2/2 ] * dx^2
    """
    gx = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
    gy = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    f  = -0.5 * phi**2 + 0.25 * phi**4 + 0.5 * (gx**2 + gy**2)
    return np.sum(f) * dx**2

# ── 4. Visualisation setup ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')

im = ax.imshow(
    phi, vmin=-1, vmax=1,
    cmap='RdBu_r', origin='lower',
    interpolation='bilinear'
)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('φ', color='white', fontsize=12)
cbar.ax.yaxis.set_tick_params(color='white')
cbar.outline.set_edgecolor('#555')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax.set_title('Cahn-Hilliard Phase Separation', color='white', fontsize=13, pad=10)
ax.set_xlabel('x', color='white')
ax.set_ylabel('y', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')

info = ax.text(
    0.02, 0.97, '', transform=ax.transAxes,
    color='white', fontsize=9, va='top',
    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
)

plt.tight_layout()
plt.pause(0.1)

# ── 5. Main simulation loop ───────────────────────────────────────────────────

print(f"Simulation start:  N={N}, dx={dx}, dt={dt}, phi0={phi0}")
print(f"Total {n_steps} steps, plot updated every {plot_interval} steps\n")

for n in range(1, n_steps + 1):

    phi = step(phi)

    if n % plot_interval == 0:
        F = free_energy(phi)
        t = n * dt

        im.set_data(phi)
        info.set_text(f'step = {n}\nt = {t:.1f}\nF = {F:.2f}')
        fig.canvas.draw()
        plt.pause(0.001)

        if n % (plot_interval * 20) == 0:
            print(f"  step {n:6d} | t = {t:7.1f} | F = {F:.4f}")

print("\nSimulation complete!")
plt.show()