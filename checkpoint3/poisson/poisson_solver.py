"""
Task 6: Poisson Equation Solver
=================================
Solves the Poisson equation in 3D:
    nabla^2 phi = -rho

using the Gauss-Seidel iterative algorithm with Dirichlet boundary
conditions (phi = 0 on all faces of the box).

Units: dx = epsilon = 1 (absorbed into rho, see Task 2).

Outputs:
    - Contour plot of phi on the midplane (z = N//2)
    - Vector plot of E on the midplane
    - phi vs r  and  |E| vs r  with power-law fits
    - Datafile: checkpoint3/poisson/data/task6_data.csv   (midplane phi, E)
                checkpoint3/poisson/data/task7_radial.csv (r, phi, |E|)
    - Plots:    checkpoint3/poisson/fig/task6_results.png
                checkpoint3/poisson/fig/task7_radial.png
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

N         = 50       # Grid size (N x N x N)
tolerance = 1e-6     # Convergence: mean|phi_new - phi_old| < tol
max_iter  = 20000    # Safety cap on iterations

# ── Charge distribution ───────────────────────────────────────────────────────

rho = np.zeros((N, N, N))
cx = cy = cz = N // 2
rho[cx, cy, cz] = 1.0      # single point charge at centre

# ── Gauss-Seidel solver (vectorised red-black scheme) ─────────────────────────

def gauss_seidel(rho, tol, max_iter):
    """
    Solve nabla^2 phi = -rho using a vectorised red-black Gauss-Seidel.

    Interior points are split into two groups by (i+j+k) % 2:
      - Red   (even): updated first using current phi values
      - Black (odd):  updated second, using the just-updated red values

    Convergence check: after both red and black updates, compare phi to
    phi_old (snapshot from the START of the iteration). Stop when
    mean|phi - phi_old| < tol.
    """
    N = rho.shape[0]
    phi = np.zeros((N, N, N))

    ii, jj, kk = np.meshgrid(
        np.arange(1, N-1),
        np.arange(1, N-1),
        np.arange(1, N-1),
        indexing='ij'
    )
    red   = (ii + jj + kk) % 2 == 0
    black = ~red

    print(f"Running Gauss-Seidel  (N={N}, tol={tol:.0e}, max_iter={max_iter})")

    for iteration in range(max_iter):

        phi_old = phi.copy()

        def nb_sum(p):
            return (
                p[2:,   1:-1, 1:-1] + p[:-2,  1:-1, 1:-1] +
                p[1:-1, 2:,   1:-1] + p[1:-1, :-2,  1:-1] +
                p[1:-1, 1:-1, 2:  ] + p[1:-1, 1:-1, :-2 ]
            )

        rho_int = rho[1:-1, 1:-1, 1:-1]

        # Red update
        nb  = nb_sum(phi)
        new = (nb + rho_int) / 6.0
        phi[1:-1, 1:-1, 1:-1] = np.where(red,   new, phi[1:-1, 1:-1, 1:-1])

        # Black update
        nb  = nb_sum(phi)
        new = (nb + rho_int) / 6.0
        phi[1:-1, 1:-1, 1:-1] = np.where(black, new, phi[1:-1, 1:-1, 1:-1])

        delta = np.mean(np.abs(phi - phi_old))

        if iteration % 500 == 0:
            print(f"  iter {iteration:6d} | delta = {delta:.3e} | phi_max = {phi.max():.5f}")

        if delta < tol:
            print(f"  Converged at iteration {iteration}  (delta = {delta:.3e})\n")
            return phi, iteration

    print(f"  Warning: did not converge within {max_iter} iterations.\n")
    return phi, max_iter

# ── Electric field ─────────────────────────────────────────────────────────────

def compute_E(phi):
    """E = -grad(phi) via centred differences (dx=1)."""
    Ex = np.zeros_like(phi)
    Ey = np.zeros_like(phi)
    Ez = np.zeros_like(phi)
    Ex[1:-1, :,    :   ] = -(phi[2:,  :,    :   ] - phi[:-2, :,    :   ]) / 2
    Ey[:,    1:-1, :   ] = -(phi[:,   2:,   :   ] - phi[:,   :-2,  :   ]) / 2
    Ez[:,    :,    1:-1] = -(phi[:,   :,    2:  ] - phi[:,   :,    :-2 ]) / 2
    return Ex, Ey, Ez

# ── Contour + vector plot ─────────────────────────────────────────────────────

def plot_results(phi, Ex, Ey, N, save_path):
    mid     = N // 2
    phi_mid = phi[:, :, mid]
    Ex_mid  = Ex[:, :, mid]
    Ey_mid  = Ey[:, :, mid]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f0f1a')

    x = np.arange(N)
    X, Y = np.meshgrid(x, x, indexing='ij')

    # Left: contour plot of phi
    ax1 = axes[0]
    ax1.set_facecolor('#0f0f1a')
    cf   = ax1.contourf(X, Y, phi_mid, levels=40, cmap='RdBu_r')
    cbar = fig.colorbar(cf, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('ϕ', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('#555')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    ax1.set_title(f'Potential ϕ  (midplane z={mid})', color='white', fontsize=13)
    ax1.set_xlabel('x', color='white');  ax1.set_ylabel('y', color='white')
    ax1.tick_params(colors='white')
    for sp in ax1.spines.values(): sp.set_edgecolor('#444')

    # Right: vector plot of E
    ax2 = axes[1]
    ax2.set_facecolor('#0f0f1a')
    skip = 3
    Xs = X[::skip, ::skip];  Ys = Y[::skip, ::skip]
    Us = Ex_mid[::skip, ::skip];  Vs = Ey_mid[::skip, ::skip]
    mag = np.sqrt(Us**2 + Vs**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        Un = np.where(mag > 0, Us / mag, 0.0)
        Vn = np.where(mag > 0, Vs / mag, 0.0)
    log_mag = np.log10(mag + 1e-10)
    qv = ax2.quiver(Xs, Ys, Un, Vn, log_mag, cmap='plasma', alpha=0.9, angles='xy')
    cb2 = fig.colorbar(qv, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label('log₁₀|E|', color='white', fontsize=11)
    cb2.ax.yaxis.set_tick_params(color='white')
    cb2.outline.set_edgecolor('#555')
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')
    ax2.set_title(f'Electric Field E  (midplane z={mid})', color='white', fontsize=13)
    ax2.set_xlabel('x', color='white');  ax2.set_ylabel('y', color='white')
    ax2.tick_params(colors='white')
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')

    plt.suptitle('Poisson Equation — Task 6', color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f"Plot saved: {save_path}")
    plt.show()

# ── Radial analysis + fit ─────────────────────────────────────────────────────

def plot_radial(phi, Ex, Ey, Ez, N, plot_path, csv_path):
    """
    Fit phi ~ A * r^alpha and |E| ~ B * r^beta.
    Saves the radial data (r, phi, |E|) to csv_path.
    """
    centre = N // 2

    ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N),
                              indexing='ij')
    r     = np.sqrt((ii - centre)**2 + (jj - centre)**2 + (kk - centre)**2)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    r_flat    = r.ravel()
    phi_flat  = phi.ravel()
    Emag_flat = E_mag.ravel()

    r_min, r_max = 2.0, 10.0
    mask = (r_flat > r_min) & (r_flat < r_max)

    r_fit    = r_flat[mask]
    phi_fit  = phi_flat[mask]
    Emag_fit = Emag_flat[mask]

    # ── Save radial CSV ───────────────────────────────────────────────────────
    # Sort by r for readability
    sort_idx = np.argsort(r_fit)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r", "phi", "E_mag"])
        for i in sort_idx:
            writer.writerow([f"{r_fit[i]:.4f}",
                             f"{phi_fit[i]:.6f}",
                             f"{Emag_fit[i]:.6f}"])
    print(f"Radial data saved: {csv_path}")

    # ── Fit ───────────────────────────────────────────────────────────────────
    def power_law(r, A, alpha):
        return A * r**alpha

    popt_phi, _ = curve_fit(power_law, r_fit, phi_fit,  p0=[0.1, -1.0], maxfev=5000)
    popt_E,   _ = curve_fit(power_law, r_fit, Emag_fit, p0=[0.1, -2.0], maxfev=5000)

    A_phi, alpha_phi = popt_phi
    A_E,   alpha_E   = popt_E

    print(f"Fit:  phi  = {A_phi:.4f} * r^{alpha_phi:.3f}   (theory: exponent = -1)")
    print(f"Fit:  |E|  = {A_E:.4f}  * r^{alpha_E:.3f}   (theory: exponent = -2)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    r_curve = np.linspace(r_min, r_max, 200)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f0f1a')

    for ax, r_data, y_data, popt, ylabel, title in zip(
        axes,
        [r_fit,    r_fit],
        [phi_fit,  Emag_fit],
        [popt_phi, popt_E],
        ['ϕ',      '|E|'],
        ['Potential ϕ vs r', 'Electric field |E| vs r'],
    ):
        ax.set_facecolor('#0f0f1a')

        idx = np.random.choice(len(r_data), size=min(3000, len(r_data)), replace=False)
        ax.scatter(r_data[idx], y_data[idx],
                   s=3, alpha=0.3, color='#4cc9f0', label='Numerical data')

        A, alpha = popt
        ax.plot(r_curve, power_law(r_curve, A, alpha),
                color='#f4a261', linewidth=2,
                label=f'Fit: {A:.3f} · r^{alpha:.3f}')

        ax.set_xlabel('r  (lattice units)', color='white', fontsize=11)
        ax.set_ylabel(ylabel, color='white', fontsize=11)
        ax.set_title(title, color='white', fontsize=13)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor('#444')

    plt.suptitle('Radial Dependence — Point Charge (Task 7)',
                 color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f"Plot saved: {plot_path}")
    plt.show()

# ── Save midplane datafile (Task 6) ──────────────────────────────────────────

def save_csv(phi, Ex, Ey, Ez, N, csv_path):
    """Save phi and E on the midplane (z = N//2) to CSV."""
    mid = N // 2
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "phi", "Ex", "Ey", "Ez"])
        for i in range(N):
            for j in range(N):
                writer.writerow([i, j, mid,
                                  f"{phi[i,j,mid]:.6f}",
                                  f"{Ex[i,j,mid]:.6f}",
                                  f"{Ey[i,j,mid]:.6f}",
                                  f"{Ez[i,j,mid]:.6f}"])
    print(f"Data saved: {csv_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

phi, n_iter = gauss_seidel(rho, tolerance, max_iter)
Ex, Ey, Ez  = compute_E(phi)

plot_results(phi, Ex, Ey, N,
             save_path=os.path.join(FIG_DIR, "task6_results.png"))

plot_radial(phi, Ex, Ey, Ez, N,
            plot_path=os.path.join(FIG_DIR,  "task7_radial.png"),
            csv_path =os.path.join(DATA_DIR, "task7_radial.csv"))

save_csv(phi, Ex, Ey, Ez, N,
         csv_path=os.path.join(DATA_DIR, "task6_data.csv"))

print(f"\nDone.  N={N},  converged in {n_iter} iterations.")