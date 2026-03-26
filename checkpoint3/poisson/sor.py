"""
Task 10: Successive Over-Relaxation (SOR)
==========================================
Finds the optimal relaxation parameter omega that minimises the number
of iterations needed to solve the 2D Poisson equation (magnetic wire
problem from Task 9).

SOR update rule:
    Az_new = (1 - omega) * Az_old + omega * Az_GS

where Az_GS is the standard Gauss-Seidel estimate.

    omega = 1          ->  standard Gauss-Seidel
    1 < omega < 2      ->  over-relaxation (faster convergence)
    omega >= 2         ->  diverges

Output:
    - checkpoint3/poisson/fig/task10_sor.png
    - checkpoint3/poisson/data/task10_sor.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# ── Save paths ────────────────────────────────────────────────────────────────

FIG_DIR  = "checkpoint3/poisson/fig"
DATA_DIR = "checkpoint3/poisson/data"
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Problem setup (same as Task 9) ────────────────────────────────────────────

N         = 50       # Smaller grid for fast omega sweep (same physics)
tolerance = 1e-4     # Looser tolerance to keep runtime short
max_iter  = 20000

Jz = np.zeros((N, N))
Jz[N//2, N//2] = 1.0   # single wire at centre

# ── SOR solver ────────────────────────────────────────────────────────────────

def sor_2d(source, omega, tol, max_iter):
    """
    Solve nabla^2 Az = -source using SOR (red-black scheme).

    Each site is updated as:
        Az_GS  = (nb_sum + source) / 4          <- Gauss-Seidel estimate
        Az_new = (1 - omega) * Az + omega * Az_GS   <- SOR blend

    omega = 1 recovers standard Gauss-Seidel.
    Returns the number of iterations to convergence.
    """
    N = source.shape[0]
    Az = np.zeros((N, N))

    ii, jj = np.meshgrid(np.arange(1, N-1), np.arange(1, N-1), indexing='ij')
    red   = (ii + jj) % 2 == 0
    black = ~red

    for iteration in range(max_iter):

        Az_old = Az.copy()

        def nb_sum(A):
            return (
                A[2:,   1:-1] + A[:-2,  1:-1] +
                A[1:-1, 2:  ] + A[1:-1, :-2 ]
            )

        src_int = source[1:-1, 1:-1]

        # Red update with SOR
        nb      = nb_sum(Az)
        Az_GS   = (nb + src_int) / 4.0
        Az_sor  = (1 - omega) * Az[1:-1, 1:-1] + omega * Az_GS
        Az[1:-1, 1:-1] = np.where(red,   Az_sor, Az[1:-1, 1:-1])

        # Black update with SOR (uses freshly updated red values)
        nb      = nb_sum(Az)
        Az_GS   = (nb + src_int) / 4.0
        Az_sor  = (1 - omega) * Az[1:-1, 1:-1] + omega * Az_GS
        Az[1:-1, 1:-1] = np.where(black, Az_sor, Az[1:-1, 1:-1])

        delta = np.mean(np.abs(Az - Az_old))

        if delta < tol:
            return iteration + 1    # converged

    return max_iter                 # did not converge within max_iter

# ── Sweep omega ───────────────────────────────────────────────────────────────

omega_values = np.linspace(1.0, 1.99, 60)
iter_counts  = []

print(f"Sweeping omega  (N={N}, tol={tolerance:.0e})")
print(f"{'omega':>8}   {'iterations':>12}")
print("-" * 25)

for omega in omega_values:
    n = sor_2d(Jz, omega, tolerance, max_iter)
    iter_counts.append(n)
    if abs(omega - round(omega * 10) / 10) < 0.01:   # print every ~0.1
        print(f"  omega = {omega:.3f}   ->  {n:6d} iterations")

iter_counts  = np.array(iter_counts)
best_idx     = np.argmin(iter_counts)
best_omega   = omega_values[best_idx]
best_iters   = iter_counts[best_idx]

print(f"\nOptimal omega = {best_omega:.3f}  ->  {best_iters} iterations")
print(f"Gauss-Seidel (omega=1.0) ->  {iter_counts[0]} iterations")
print(f"Speed-up factor: {iter_counts[0] / best_iters:.1f}x")

# ── Save CSV ──────────────────────────────────────────────────────────────────

csv_path = os.path.join(DATA_DIR, "task10_sor.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["omega", "iterations"])
    for omega, n in zip(omega_values, iter_counts):
        writer.writerow([f"{omega:.4f}", n])
print(f"\nData saved: {csv_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')

ax.plot(omega_values, iter_counts, color='#4cc9f0', linewidth=2)

# Mark optimal omega
ax.axvline(best_omega, color='#f4a261', linewidth=1.5, linestyle='--')
ax.scatter([best_omega], [best_iters], color='#f4a261', s=80, zorder=5,
           label=f'Optimal ω = {best_omega:.3f}  ({best_iters} iters)')

# Mark Gauss-Seidel baseline (omega = 1)
ax.axvline(1.0, color='white', linewidth=1, linestyle=':')
ax.scatter([1.0], [iter_counts[0]], color='white', s=60, zorder=5,
           label=f'Gauss-Seidel (ω=1)  ({iter_counts[0]} iters)')

ax.set_xlabel('ω  (relaxation parameter)', color='white', fontsize=12)
ax.set_ylabel('Iterations to convergence', color='white', fontsize=12)
ax.set_title('SOR Convergence vs ω  (Task 10)', color='white', fontsize=14)
ax.tick_params(colors='white')
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=10)
for sp in ax.spines.values(): sp.set_edgecolor('#444')

plt.tight_layout()
save_path = os.path.join(FIG_DIR, "task10_sor.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
print(f"Plot saved: {save_path}")
plt.show()