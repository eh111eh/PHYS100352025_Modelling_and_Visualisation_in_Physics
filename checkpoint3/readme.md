# Checkpoint 3: Partial Differential Equations

This repository contains the numerical simulations and analysis for the Cahn-Hilliard phase separation model and the Poisson equation (electrostatics and magnetostatics).

## 1. Directory Structure

```
checkpoint3/
├── cahn-hilliard/
│   ├── cahn_hilliard.py
│   ├── task3_convergence.py
│   ├── task5_free_energy.py
│   ├── fig/
│   │   ├── task3_stability.png
│   │   └── task5_free_energy.png
│   └── data/
│       ├── task3_boundary_2d.csv
│       ├── task3_boundary_dx1.csv
│       └── task5_free_energy.csv
│
└── poisson/
    ├── poisson_solver.py
    ├── magnetic_solver.py
    ├── task10_sor.py
    ├── fig/
    │   ├── task6_results.png
    │   ├── task7_radial.png
    │   ├── task9_results.png
    │   ├── task9_radial.png
    │   └── task10_sor.png
    └── data/
        ├── task6_data.csv
        ├── task7_radial.csv
        ├── task9_data.csv
        └── task10_sor.csv
```

---

## 2. Part 1: Cahn-Hilliard Equation (Section 3.1)

### Task 2: Main Simulation & Animation
**File:** `cahn-hilliard/cahn_hilliard.py`  
**Description:** Solves the dimensionless Cahn-Hilliard equation in 2D using an explicit Euler scheme with centred differences and periodic boundary conditions.

$$\frac{\partial\phi}{\partial\tilde{t}} = \tilde{\nabla}^2\tilde{\mu}, \qquad \tilde{\mu} = -\phi(1-\phi^2) - \tilde{\nabla}^2\phi$$

* **How to run:** `python cahn_hilliard.py`
* **Parameters to adjust:**
    * `phi0` — initial mean composition (`0.0` for spinodal decomposition, `±0.5` for droplet formation)
    * `dt` — time step (stable for `dt = 0.01`, `dx = 1.0`)
    * `N` — grid size (default `100×100`)
* **Features:** Real-time animation of the phase field φ(x, y). The left panel shows the composition field (blue = oil, red = water) updating live as phase separation evolves.

### Task 3: Convergence Conditions
**File:** `cahn-hilliard/task3_convergence.py`  
**Description:** Sweeps over `dt` and `dx` to map out the stable and unstable regions of parameter space. The theoretical stability bound is:

$$dt \leq \frac{dx^4}{8(dx^2 + 4)}$$

* **How to run:** `python task3_convergence.py`
* **Output:**
    * **Data:** `data/task3_boundary_2d.csv` (boundary points in dx-dt space), `data/task3_boundary_dx1.csv` (boundary points for dx=1 sweep)
    * **Graph:** `fig/task3_stability.png` (left: 2D stability map; right: dt sweep at dx=1.0)

### Task 5: Free Energy Minimisation
**File:** `cahn-hilliard/task5_free_energy.py`  
**Description:** Runs the simulation for `phi0 = 0` and `phi0 = 0.5` and tracks the dimensionless free energy over time:

$$F = \int \left[ -\frac{\phi^2}{2} + \frac{\phi^4}{4} + \frac{|\nabla\phi|^2}{2} \right] dx^2$$

* **How to run:** `python task5_free_energy.py`
* **Runtime:** ~1 minute (50000 steps × 2 cases)
* **Output:**
    * **Data:** `data/task5_free_energy.csv` (columns: `step`, `F_phi0_0`, `F_phi0_05`)
    * **Graph:** `fig/task5_free_energy.png` (F vs step for both initial conditions, with equilibrium value marked)

---

## 3. Part 2: Poisson Equation (Section 3.2)

### Tasks 6 & 7: Electric Potential and Field (Point Charge)
**File:** `poisson/poisson_solver.py`  
**Description:** Solves the 3D Poisson equation ∇²ϕ = −ρ for a single point charge at the centre of the box using a vectorised red-black Gauss-Seidel algorithm with Dirichlet boundary conditions (ϕ = 0 on all faces).

$$\phi^{n+1}_{i,j,k} = \frac{1}{6}\left(\phi^n_{i+1,j,k} + \phi^n_{i-1,j,k} + \phi^n_{i,j+1,k} + \phi^n_{i,j-1,k} + \phi^n_{i,j,k+1} + \phi^n_{i,j,k-1} + \rho_{i,j,k}\right)$$

The electric field is computed as **E = −∇ϕ** via centred differences. The radial dependence is fitted to verify Gauss's law: ϕ ∝ 1/r and |E| ∝ 1/r².

* **How to run:** `python poisson_solver.py`
* **Parameters to adjust:**
    * `N` — grid size (default `50×50×50`)
    * `tolerance` — convergence criterion (default `1e-6`)
* **Runtime:** ~2–5 minutes
* **Output:**
    * **Data:** `data/task6_data.csv` (midplane ϕ and E), `data/task7_radial.csv` (r, ϕ, |E|)
    * **Graphs:** `fig/task6_results.png` (contour plot of ϕ, vector plot of E), `fig/task7_radial.png` (ϕ vs r and |E| vs r with power-law fits)

### Tasks 9: Magnetic Vector Potential (Infinite Wire)
**File:** `poisson/magnetic_solver.py`  
**Description:** Solves the 2D Poisson equation ∇²Az = −µ₀Jz for a single wire running through the origin. Since the wire is infinite in z, the problem reduces to 2D. The magnetic field is computed as:

$$B_x = \frac{\partial A_z}{\partial y}, \qquad B_y = -\frac{\partial A_z}{\partial x}$$

The radial dependence is fitted to verify the theoretical result: Az ∝ −ln(r) and |B| ∝ 1/r.

* **How to run:** `python magnetic_solver.py`
* **Parameters to adjust:**
    * `N` — grid size (default `100×100`)
    * `tolerance` — convergence criterion (default `1e-6`)
* **Runtime:** ~1–2 minutes
* **Output:**
    * **Data:** `data/task9_data.csv` (Az, Bx, By for all grid points)
    * **Graphs:** `fig/task9_results.png` (contour plot of Az, vector plot of B), `fig/task9_radial.png` (Az vs r and |B| vs r with fits)

### Task 10: Successive Over-Relaxation (SOR)
**File:** `poisson/task10_sor.py`  
**Description:** Extends the Gauss-Seidel solver with a relaxation parameter ω:

$$A_z^{n+1} = (1-\omega)A_z^n + \omega A_z^{GS}$$

Sweeps ω from 1.0 to 1.99 to find the value that minimises the number of iterations to convergence. ω = 1 recovers standard Gauss-Seidel.

* **How to run:** `python task10_sor.py`
* **Runtime:** ~1–2 minutes (60 values of ω × ~300 iterations each)
* **Output:**
    * **Data:** `data/task10_sor.csv` (columns: `omega`, `iterations`)
    * **Graph:** `fig/task10_sor.png` (iterations vs ω, with optimal ω marked)

---

## 4. Requirements

* Python 3.x
* NumPy
* Matplotlib
* SciPy (`curve_fit` used in `poisson_solver.py` and `magnetic_solver.py`)