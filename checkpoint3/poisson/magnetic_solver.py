"""
Task 9: Magnetic Vector Potential — Infinite Wire
===================================================
Solves the 2D Poisson equation for the magnetic vector potential Az:
    nabla^2_xy Az = -mu0 * Jz(x, y)

Since the wire is infinite in z, the problem reduces to 2D (x, y only).
The magnetic field is then:
    Bx =  dAz/dy
    By = -dAz/dx
    Bz =  0

Units: dx = mu0 = 1 (absorbed into Jz, same reasoning as Task 2).

Theory (infinite wire):
    Az(r) ~ -A * ln(r) + C       (logarithmic)
    |B|(r) ~ A / r               (1/r)

Outputs:
    - Contour plot of Az and vector plot of B
    - Az vs r and |B| vs r with fits
    - checkpoint3/magnetic/fig/task9_results.png
    - checkpoint3/magnetic/fig/task9_radial.png
    - checkpoint3/magnetic/data/task9_data.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import csv

# ── Save paths ────────────────────────────────────────────────────────────────

FIG_DIR  = "checkpoint3/poisson/fig"
DATA_DIR = "checkpoint3/poisson/data"
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── User settings ─────────────────────────────────────────────────────────────

N         = 100      # Grid size (N x N),  2D problem so 100x100 is fast
tolerance = 1e-6     # Convergence: mean|Az_new - Az_old| < tol
max_iter  = 20000    # Safety cap on iterations

# ── Current distribution: single wire at centre ───────────────────────────────

Jz = np.zeros((N, N))
cx = cy = N // 2
Jz[cx, cy] = 1.0     # unit current at centre (mu0 = 1 absorbed)

# ── Gauss-Seidel solver (2D, vectorised red-black) ────────────────────────────

def gauss_seidel_2d(source, tol, max_iter):
    """
    Solve nabla^2 Az = -source in 2D using red-black Gauss-Seidel.

    Update rule (centred difference, dx=1):
        Az[i,j] = (Az[i+1,j] + Az[i-1,j] + Az[i,j+1] + Az[i,j-1]
                   + source[i,j]) / 4

    Boundary conditions: Az = 0 on all edges (Dirichlet).
    Convergence: stop when mean|Az - Az_old| < tol.
    """
    N = source.shape[0]
    Az = np.zeros((N, N))

    # Red-black masks for interior points
    ii, jj = np.meshgrid(np.arange(1, N-1), np.arange(1, N-1), indexing='ij')
    red   = (ii + jj) % 2 == 0
    black = ~red

    print(f"Running 2D Gauss-Seidel  (N={N}, tol={tol:.0e}, max_iter={max_iter})")

    for iteration in range(max_iter):

        Az_old = Az.copy()

        def nb_sum(A):
            return (
                A[2:,   1:-1] + A[:-2,  1:-1] +
                A[1:-1, 2:  ] + A[1:-1, :-2 ]
            )

        src_int = source[1:-1, 1:-1]

        # Red update
        nb  = nb_sum(Az)
        new = (nb + src_int) / 4.0
        Az[1:-1, 1:-1] = np.where(red,   new, Az[1:-1, 1:-1])

        # Black update (uses freshly updated red values)
        nb  = nb_sum(Az)
        new = (nb + src_int) / 4.0
        Az[1:-1, 1:-1] = np.where(black, new, Az[1:-1, 1:-1])

        delta = np.mean(np.abs(Az - Az_old))

        if iteration % 500 == 0:
            print(f"  iter {iteration:6d} | delta = {delta:.3e} | Az_max = {Az.max():.5f}")

        if delta < tol:
            print(f"  Converged at iteration {iteration}  (delta = {delta:.3e})\n")
            return Az, iteration

    print(f"  Warning: did not converge within {max_iter} iterations.\n")
    return Az, max_iter

# ── Magnetic field B = curl(A) ────────────────────────────────────────────────

def compute_B(Az):
    """
    Bx =  dAz/dy   (centred difference)
    By = -dAz/dx   (centred difference)
    Bz =  0        (wires all in z direction)
    """
    Bx = np.zeros_like(Az)
    By = np.zeros_like(Az)

    Bx[:, 1:-1] =  (Az[:, 2:] - Az[:, :-2]) / 2    #  dAz/dy
    By[1:-1, :] = -(Az[2:, :] - Az[:-2, :]) / 2    # -dAz/dx

    return Bx, By

# ── Contour + vector plot ─────────────────────────────────────────────────────

def plot_results(Az, Bx, By, N, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f0f1a')

    x = np.arange(N)
    X, Y = np.meshgrid(x, x, indexing='ij')

    # ── Left: contour plot of Az ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#0f0f1a')
    cf   = ax1.contourf(X, Y, Az, levels=40, cmap='RdBu_r')
    cbar = fig.colorbar(cf, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Az', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('#555')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax1.set_title('Vector Potential Az', color='white', fontsize=13)
    ax1.set_xlabel('x', color='white');  ax1.set_ylabel('y', color='white')
    ax1.tick_params(colors='white')
    for sp in ax1.spines.values(): sp.set_edgecolor('#444')

    # ── Right: vector plot of B ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#0f0f1a')

    skip = 5
    Xs = X[::skip, ::skip];  Ys = Y[::skip, ::skip]
    Us = Bx[::skip, ::skip];  Vs = By[::skip, ::skip]
    mag = np.sqrt(Us**2 + Vs**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        Un = np.where(mag > 0, Us / mag, 0.0)
        Vn = np.where(mag > 0, Vs / mag, 0.0)
    log_mag = np.log10(mag + 1e-10)
    qv = ax2.quiver(Xs, Ys, Un, Vn, log_mag, cmap='plasma', alpha=0.9, angles='xy')
    cb2 = fig.colorbar(qv, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label('log₁₀|B|', color='white', fontsize=11)
    cb2.ax.yaxis.set_tick_params(color='white')
    cb2.outline.set_edgecolor('#555')
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')
    ax2.set_title('Magnetic Field B', color='white', fontsize=13)
    ax2.set_xlabel('x', color='white');  ax2.set_ylabel('y', color='white')
    ax2.tick_params(colors='white')
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')

    plt.suptitle('Magnetic Vector Potential — Task 9', color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f"Plot saved: {save_path}")
    plt.show()

# ── Radial analysis + fit ─────────────────────────────────────────────────────

def plot_radial(Az, Bx, By, N, save_path):
    """
    Fit:
        Az  ~ -A * ln(r) + C    (theory for infinite wire)
        |B| ~  A / r            (theory for infinite wire)

    Fit range: r_min=2 to r_max=10 to avoid lattice artefacts near
    the wire and Dirichlet boundary effects far from it.
    """
    centre = N // 2

    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    r = np.sqrt((ii - centre)**2 + (jj - centre)**2)

    B_mag = np.sqrt(Bx**2 + By**2)

    r_flat    = r.ravel()
    Az_flat   = Az.ravel()
    Bmag_flat = B_mag.ravel()

    r_min, r_max = 2.0, 10.0
    mask = (r_flat > r_min) & (r_flat < r_max)

    r_fit    = r_flat[mask]
    Az_fit   = Az_flat[mask]
    Bmag_fit = Bmag_flat[mask]

    # Fit functions
    def log_fit(r, A, C):
        return -A * np.log(r) + C      # Az ~ -A*ln(r) + C

    def inv_fit(r, A):
        return A / r                   # |B| ~ A/r

    popt_Az, _ = curve_fit(log_fit, r_fit, Az_fit,   p0=[0.1, 0.5])
    popt_B,  _ = curve_fit(inv_fit, r_fit, Bmag_fit, p0=[0.1])

    A_Az, C_Az = popt_Az
    A_B        = popt_B[0]

    print(f"Fit:  Az  = -{A_Az:.4f} * ln(r) + {C_Az:.4f}   (theory: -A*ln(r))")
    print(f"Fit:  |B| =  {A_B:.4f} / r                     (theory:  A/r = mu0*I/(2pi*r))")
    print(f"      theory A = mu0*I/(2*pi) = 1/(2*pi) = {1/(2*np.pi):.4f}")

    r_curve = np.linspace(r_min, r_max, 200)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f0f1a')

    for ax, r_data, y_data, fit_fn, popt, ylabel, title, theory_label in zip(
        axes,
        [r_fit,   r_fit],
        [Az_fit,  Bmag_fit],
        [log_fit, inv_fit],
        [popt_Az, popt_B],
        ['Az',    '|B|'],
        ['Vector Potential Az vs r', 'Magnetic Field |B| vs r'],
        [f'-{A_Az:.3f}·ln(r) + {C_Az:.3f}',
         f'{A_B:.3f} / r']
    ):
        ax.set_facecolor('#0f0f1a')

        idx = np.random.choice(len(r_data), size=min(3000, len(r_data)), replace=False)
        ax.scatter(r_data[idx], y_data[idx],
                   s=3, alpha=0.3, color='#4cc9f0', label='Numerical data')

        ax.plot(r_curve, fit_fn(r_curve, *popt),
                color='#f4a261', linewidth=2, label=f'Fit: {theory_label}')

        ax.set_xlabel('r  (lattice units)', color='white', fontsize=11)
        ax.set_ylabel(ylabel, color='white', fontsize=11)
        ax.set_title(title, color='white', fontsize=13)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#444')

    plt.suptitle('Radial Dependence — Infinite Wire (Task 9)',
                 color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f"Plot saved: {save_path}")
    plt.show()

# ── Save datafile ─────────────────────────────────────────────────────────────

def save_csv(Az, Bx, By, N, csv_path):
    """Save Az and B for all grid points to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "Az", "Bx", "By"])
        for i in range(N):
            for j in range(N):
                writer.writerow([i, j,
                                  f"{Az[i,j]:.6f}",
                                  f"{Bx[i,j]:.6f}",
                                  f"{By[i,j]:.6f}"])
    print(f"Data saved: {csv_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

Az, n_iter = gauss_seidel_2d(Jz, tolerance, max_iter)
Bx, By     = compute_B(Az)

plot_results(Az, Bx, By, N,
             save_path=os.path.join(FIG_DIR, "task9_results.png"))

plot_radial(Az, Bx, By, N,
            save_path=os.path.join(FIG_DIR, "task9_radial.png"))

save_csv(Az, Bx, By, N,
         csv_path=os.path.join(DATA_DIR, "task9_data.csv"))

print(f"\nDone.  N={N},  converged in {n_iter} iterations.")