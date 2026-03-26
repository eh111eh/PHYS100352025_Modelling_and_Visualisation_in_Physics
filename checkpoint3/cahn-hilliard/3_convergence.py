"""
Task 3: Convergence condition experiment + visualisation
=========================================================
Sweeps over dt and dx to determine when the algorithm diverges,
and saves the results as a plot.

Output images:  checkpoint3/cahn-hilliard/fig/task3_stability.png
Output CSV:     checkpoint3/cahn-hilliard/data/task3_boundary_2d.csv
                checkpoint3/cahn-hilliard/data/task3_boundary_dx1.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import csv

# ── Save path ─────────────────────────────────────────────────────────────────

SAVE_DIR = "checkpoint3/cahn-hilliard"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Core functions ────────────────────────────────────────────────────────────

def laplacian(f, dx):
    return (
        np.roll(f,  1, axis=0) + np.roll(f, -1, axis=0) +
        np.roll(f,  1, axis=1) + np.roll(f, -1, axis=1) -
        4 * f
    ) / dx**2

def step(phi, dx, dt):
    mu = -phi * (1 - phi**2) - laplacian(phi, dx)
    return phi + dt * laplacian(mu, dx)

def is_stable(dx, dt, n_test=300, N=30):
    """
    Run n_test steps and return True (stable) if phi remains finite
    and within [-10, 10]. N=30 is used for a fast parameter sweep.
    """
    np.random.seed(0)
    phi = 0.01 * np.random.randn(N, N)
    for _ in range(n_test):
        phi = step(phi, dx, dt)
        if not np.isfinite(phi).all() or np.max(np.abs(phi)) > 10:
            return False
    return True

# ── Parameter sweep ───────────────────────────────────────────────────────────

dx_values = np.linspace(0.4, 2.5, 30)
dt_values = np.linspace(0.001, 0.12, 30)

print("Scanning stability... (30 dx x 30 dt = 900 combinations)")

stability = np.zeros((len(dt_values), len(dx_values)), dtype=bool)

for i, dx in enumerate(dx_values):
    for j, dt in enumerate(dt_values):
        stability[j, i] = is_stable(dx, dt)
    print(f"  dx = {dx:.2f} done")

# ── Extract boundary points (left panel: 2D map) ──────────────────────────────
# A point (dx, dt) is on the boundary if it is stable but has at least one
# unstable neighbour (or vice versa) in the dx or dt direction.

boundary_2d = []   # list of (dx, dt, stable) tuples at the boundary

for j, dt in enumerate(dt_values):
    for i, dx in enumerate(dx_values):
        current = stability[j, i]
        # check 4-connected neighbours
        neighbours = []
        if i > 0:               neighbours.append(stability[j, i-1])
        if i < len(dx_values)-1:neighbours.append(stability[j, i+1])
        if j > 0:               neighbours.append(stability[j-1, i])
        if j < len(dt_values)-1:neighbours.append(stability[j+1, i])
        # boundary = current cell differs from at least one neighbour
        if any(n != current for n in neighbours):
            boundary_2d.append((dx, dt, int(current)))

csv_path_2d = os.path.join(SAVE_DIR, "data/task3_boundary_2d.csv")
with open(csv_path_2d, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dx", "dt", "stable"])   # 1 = stable, 0 = unstable
    writer.writerows(boundary_2d)
print(f"\nBoundary points (2D map) saved: {csv_path_2d}  ({len(boundary_2d)} points)")

# ── Extract boundary points (right panel: dx=1 sweep) ────────────────────────
# For the 1D sweep, the boundary is every (dt, stable) pair where the
# stability status changes between consecutive dt values.

dt_scan  = np.linspace(0.001, 0.06, 60)
dx_fixed = 1.0
results  = [is_stable(dx_fixed, dt, n_test=500) for dt in dt_scan]

boundary_dx1 = []
for k in range(len(dt_scan)):
    current = results[k]
    # boundary = status differs from the next point
    if k < len(dt_scan) - 1 and results[k+1] != current:
        boundary_dx1.append((dt_scan[k],   int(current)))
        boundary_dx1.append((dt_scan[k+1], int(results[k+1])))

csv_path_dx1 = os.path.join(SAVE_DIR, "data/task3_boundary_dx1.csv")
with open(csv_path_dx1, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dt", "stable"])         # 1 = stable, 0 = unstable
    writer.writerows(boundary_dx1)
print(f"Boundary points (dx=1 sweep) saved: {csv_path_dx1}  ({len(boundary_dx1)} points)")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0f0f1a')

# ── Left panel: 2D stability map (dx vs dt) ───────────────────────────────────

ax = axes[0]
ax.set_facecolor('#0f0f1a')

DX, DT = np.meshgrid(dx_values, dt_values)
ax.contourf(DX, DT, stability.astype(float), levels=[-0.5, 0.5, 1.5],
            colors=['#e63946', '#457b9d'], alpha=0.7)

# Overlay boundary points from CSV
bdx = [p[0] for p in boundary_2d]
bdt = [p[1] for p in boundary_2d]
ax.scatter(bdx, bdt, color='white', s=12, zorder=4, label='Boundary points (CSV)')

ax.axvline(x=1.0, color='#f4a261', linewidth=1, linestyle=':')
ax.axhline(y=0.01, color='#f4a261', linewidth=1, linestyle=':')
ax.plot(1.0, 0.01, 'o', color='#f4a261', markersize=8, zorder=5)

ax.set_xlabel('dx  (spatial step)', color='white', fontsize=11)
ax.set_ylabel('dt  (time step)', color='white', fontsize=11)
ax.set_title('Stability map: dt vs dx', color='white', fontsize=13)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')

stable_patch   = mpatches.Patch(color='#457b9d', alpha=0.7, label='Stable (converges)')
unstable_patch = mpatches.Patch(color='#e63946', alpha=0.7, label='Unstable (diverges)')
current_point  = plt.Line2D([0],[0], marker='o', color='#f4a261', linestyle='',
                             label='Current code (dx=1, dt=0.01)')
boundary_dot   = plt.Line2D([0],[0], marker='o', color='white', linestyle='',
                             markersize=4, label='Boundary points (CSV)')
ax.legend(handles=[stable_patch, unstable_patch, current_point, boundary_dot],
          facecolor='#1a1a2e', labelcolor='white', fontsize=8)

# ── Right panel: fixed dx=1.0, sweep over dt ──────────────────────────────────

ax2 = axes[1]
ax2.set_facecolor('#0f0f1a')

bar_colors = ['#457b9d' if r else '#e63946' for r in results]
ax2.bar(dt_scan, [1]*len(dt_scan), width=dt_scan[1]-dt_scan[0],
        color=bar_colors, align='edge', alpha=0.8)

# Mark boundary dt values as vertical white lines
for dt_b, _ in boundary_dx1:
    ax2.axvline(x=dt_b, color='white', linewidth=1.2, linestyle='--', alpha=0.8)

ax2.set_xlabel('dt  (time step, dx=1.0 fixed)', color='white', fontsize=11)
ax2.set_title('Convergence vs dt  (dx = 1.0)', color='white', fontsize=13)
ax2.set_yticks([])
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

stable_patch2   = mpatches.Patch(color='#457b9d', alpha=0.8, label='Stable')
unstable_patch2 = mpatches.Patch(color='#e63946', alpha=0.8, label='Unstable')
boundary_line   = plt.Line2D([0],[0], color='white', linestyle='--',
                              label='Boundary dt (CSV)')
ax2.legend(handles=[stable_patch2, unstable_patch2, boundary_line],
           facecolor='#1a1a2e', labelcolor='white', fontsize=9)

plt.suptitle('Cahn-Hilliard Stability Conditions (Task 3)', color='white', fontsize=14, y=1.01)
plt.tight_layout()

save_path = os.path.join(SAVE_DIR, "task3_stability.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
print(f"Plot saved: {save_path}")
plt.show()