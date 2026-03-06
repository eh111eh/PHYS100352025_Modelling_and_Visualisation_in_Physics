# Checkpoint 1: Monte Carlo Simulations of the 2D Ising Model

This repository contains the implementation of the 2D Ising Model using **Glauber** and **Kawasaki** dynamics. The project explores phase transitions, thermodynamic observables, and the effects of different Markov Chain Monte Carlo (MCMC) update rules.

---

## 1. Project Structure
The code is designed to be self-organizing. Upon running the scripts, the following directories will be created:
* `checkpoint1/data/`: Stores `.txt` datafiles for quantitative analysis.
* `checkpoint1/figures/`: Stores `.png` plots and `.gif` animations.

---

## 2. Glauber Dynamics (Tasks 1–5)
Glauber dynamics uses a single spin-flip algorithm where magnetization is not conserved.

### Scripts
* **`ising_glauber.py`**: Visual simulation of the Ising model.
  * **Usage**: `python ising_glauber.py`
  * **Output**: Displays a live animation and saves `glauber_animation.gif`.
* **`magnetisation.py`**: Measures average absolute magnetization $\langle |M| \rangle$ and susceptibility $\chi$.
  * **Output**: Saves `task4_glauber_data.txt` and `magnetisation_plots.png`.
* **`ising_energy_cv.py`**: Measures average energy $\langle E \rangle$ and specific heat $C_v$.
  * **Error Analysis**: Uses **Bootstrap resampling** to calculate error bars for $C_v$.
  * **Output**: Saves `task5_numba_data.txt` and `energy_cv_plots.png`.

### Physical Analysis
As the system approaches the critical temperature $T_c \approx 2.27$ $J/k_B$, we observe a peak in both susceptibility and specific heat. This indicates a second-order phase transition from an ordered ferromagnetic state to a disordered paramagnetic state.

---

## 3. Kawasaki Dynamics (Task 6)
Kawasaki dynamics utilizes a spin-exchange algorithm.

### Scripts
* **`ising_kawasaki.py`**: Visual simulation of the exchange process.
  * **Usage**: `python ising_kawasaki.py`
  * **Output**: Saves `kawasaki_animation.gif`.
* **`ising_kawasaki_thermal.py`**: Quantitative analysis of thermal properties.
  * **Output**: Saves `task6_kawasaki_data.txt` and `task6_kawasaki_plots.png`.

### Why Magnetization ($M$) is not used:
In Kawasaki dynamics, spins are exchanged rather than flipped. This means the total number of $+1$ and $-1$ spins is constant. Consequently, the magnetization $M$ is a **conserved quantity**. If initialized at $M \approx 0$, it will remain there regardless of temperature, providing no information about the phase transition. Instead, we use the **Total Energy** and **Specific Heat** to observe the spatial ordering (clustering) of the spins.

### 3.1 Interactive CLI Tool: `ising_simulation.py`
This script serves as a universal, interactive visualization tool for the Ising model. Unlike the task-specific scripts, this file uses command-line arguments to allow for real-time comparisons between dynamics. 

It is designed for use during in-person marking to demonstrate:
* **Glauber Dynamics (--dynamics glauber):** To show "critical opalescence" and the formation/breaking of large clusters near $T_c \approx 2.27$.
* **Kawasaki Dynamics (--dynamics kawasaki):** To provide visual proof that magnetization is conserved. You can observe how spins "clump" and phase-separate over time while the total balance of red and blue pixels remains identical to the initial state.

**Usage Example:**
```bash
python ising_simulation.py --L 50 --T 1.5 --dynamics kawasaki
```


---

## 4. Quantitative Analysis & Error Estimation
All quantitative scripts account for:
1. **Equilibration**: A set number of sweeps are discarded to ensure the system has reached steady-state.
2. **Decorrelation**: Measurements are taken every 10–20 sweeps to ensure samples are statistically independent.
3. **Resampling (Bootstrap)**: Specific heat error bars are calculated by resampling the energy data 200 times with replacement to estimate the variance of the variance.

---

## 5. Requirements
To run these codes, you need:
* **Python 3.x**
* **NumPy** & **Matplotlib**
* **Numba**: For Just-In-Time (JIT) compilation to speed up the Metropolis loops.
* **Pillow**: Required by Matplotlib to save animations as `.gif`.

---

## 6. How to Run
To generate the full dataset for a task (e.g., Task 6), navigate to the root directory and run:
```bash
python ising_kawasaki_thermal.py