# Checkpoint 2: Cellular Automata

This repository contains the numerical simulations and analysis for Conway's Game of Life and the stochastic SIRS epidemic model.

## 1. Directory Structure
The workspace is organised as follows:
* **Root**: All `.py` script files.
* **`data/`**: Destination for generated `.csv`, `.dat`, and `.txt` files.
* **`fig/`**: Destination for saved `.png` plots.

---

## 2. Part 1: Conway’s Game of Life (Section 2.1)

### Task 1: Main Simulation & Animation
**File:** `game_of_life_sim.py`  
**Description:** Implements the deterministic 8-neighbor rules with Periodic Boundary Conditions (PBC).
* **How to run:** `python game_of_life_sim.py`
* **Features:** Choose between `Random`, `Oscillator (Blinker)`, or `Glider` via terminal input to view the live animation.

### Task 2: Equilibration Time Distribution
**File:** `equilbrium.py`  
**Description:** Measures the time steps required for a random initial state to reach a steady state (absorbing or oscillating) across 1000 trials.
* **How to run:** `python equilbrium.py`
* **Output:**
    * **Data:** `data/equilibrium_times.dat`
    * **Graph:** `fig/equilibrium_distribution.png` (Histogram of convergence times).

### Task 3: Glider Centre of Mass & Speed
**File:** `glider_com.py`  
**Description:** Tracks the Centre of Mass (CoM) of a glider. Uses **coordinate unwrapping** to handle Periodic Boundary Conditions correctly.
* **How to run:** `python glider_com.py`
* **Output:** * **Data:** `data/glider_displacement.csv`
    * **Graph:** `fig/glider_true_velocity_analysis.png` (Linear fit used to estimate $v \approx 0.35$ cells/step).

---

## 3. Part 2: SIRS Model for Epidemics (Section 2.2)

### Task 1: Random Sequential Update Simulation
**File:** `1_sirs_model.py`  
**Description:** Core simulation using the random sequential updating scheme (one site per step, $N$ steps per sweep).
* **How to run:** `python 1_sirs_model.py -size 50 -P_SI 0.5 -P_IR 0.2 -P_RS 0.05`

### Task 2: Dynamics Scenarios
**File:** `2_sirs_three_scenarios.py`  
**Description:** Visualises the three qualitative regimes: Absorbing state, Dynamic Equilibrium, and Cyclic Waves.
* **How to run:** `python 2_sirs_three_scenarios.py`
* **Output:** A side-by-side animation of the three different parameter sets.

### Task 3: Phase Diagram (Heatmap)
**File:** `3_infected_sites_heatmap.py`  
**Description:** Generates a heatmap of the average infected fraction over the $p_{S \to I} - p_{R \to S}$ plane.
* **How to run:** `python 3_infected_sites_heatmap.py`
* **Output:** * **Data:** `data/heatmap_data.txt`
    * **Graph:** `fig/sirs_heatmap.png` (Resolution: 0.05 steps).

### Task 4: Variance Analysis with Error Bars
**File:** `4_sirs_variance_analysis.py`  
**Description:** Calculates the variance $\chi = (\langle I^2 \rangle - \langle I \rangle^2)/N$ along a cut at $p_{R \to S} = 0.5$.
* **How to run:** `python 4_sirs_variance_analysis.py`
* **Error Method:** Implements **Jackknife Resampling** (20 blocks) to provide robust error estimates for the non-linear variance statistic.
* **Output:** * **Data:** `data/variance_jacknife_data.txt`
    * **Graph:** `fig/sirs_variance_jacknife.png`.

### Task 5: Vaccination Analysis
**File:** `5_sirs_vaccination_analysis.py`  
**Description:** Introduces a fraction $f_{Im}$ of permanently immune (R) sites. 
* **How to run:** `python 5_sirs_vaccination_analysis.py`
* **Output:** * **Data:** `data/vaccination_data.txt`
    * **Graph:** `fig/vaccination_effect.png` (Infected fraction vs. Immunity fraction).

---

## 4. Requirements
* Python 3.x
* NumPy
* Matplotlib