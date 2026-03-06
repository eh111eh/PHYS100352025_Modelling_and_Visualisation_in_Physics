import numpy as np
import matplotlib.pyplot as plt
import os

def update_sirs(grid, p1, p2, p3):
    S, I, R = 0, 1, 2
    L = grid.shape[0]
    new_grid = grid.copy()

    up = np.roll(grid, 1, axis=0)
    down = np.roll(grid, -1, axis=0)
    left = np.roll(grid, 1, axis=1)
    right = np.roll(grid, -1, axis=1)

    has_infected_neighbour = (up == I) | (down == I) | (left == I) | (right == I)

    mask_S_to_I = (grid == S) & has_infected_neighbour & (np.random.rand(L, L) < p1)
    mask_I_to_R = (grid == I) & (np.random.rand(L, L) < p2)
    mask_R_to_S = (grid == R) & (np.random.rand(L, L) < p3)

    new_grid[mask_S_to_I] = I
    new_grid[mask_I_to_R] = R
    new_grid[mask_R_to_S] = S

    return new_grid

def get_variance_jackknife(p1, p2, p3, size=50, eq_sweeps=500, run_sweeps=10000):
    grid = np.random.choice([0, 1, 2], size=(size, size))
    L_site = size
    N_sites = size * size

    # 1. Equilibration
    # Run 500 (eq_sweeps) pre-simulations before collecting data to ensure 
    # the initial randomness doesn't bias the system's statistical properties.
    for _ in range(eq_sweeps):
        grid = update_sirs(grid, p1, p2, p3)

    # 2. Measurement
    # Start the main simulation immediately after equilibrium is reached.
    # Run one simulation step (update_sirs) -> count the current number of infected individuals -> record it in the list (N_I).
    N_I = []
    for _ in range(run_sweeps):
        grid = update_sirs(grid, p1, p2, p3)
        N_I.append(np.count_nonzero(grid == 1))
    
    N_I = np.array(N_I, dtype=float)
    n = len(N_I)
    
    # 3. Calculate the true variance
    # C = (<I^2> - <I>^2) / L^2
    c_true = (np.mean(N_I**2) - np.mean(N_I)**2) / (L_site * L_site)

    # 4. Jackknife Procedure
    # Calculate the error by observing how the statistics change when data points are removed one by one
    sum_I = np.sum(N_I)
    sum_I2 = np.sum(N_I**2)

    mean_resample = (sum_I - N_I) / (n - 1)
    mean_resample_sq = (sum_I2 - N_I**2) / (n - 1)

    # A collection of jacknife replicates (pseudo-values) calculated by excluding the ith data point.
    ci = (mean_resample_sq - mean_resample**2) / (L_site * L_site)

    # Calculate the standard deviation of the jacknife estimates.
    # Sum the differences between the results obtained by excluding the ith data point and the original c_{true}.
    # sqrt(sum((ci - c_true)**2))
    error = np.sqrt(np.sum((ci - c_true)**2))
    
    return c_true, error

def run_variance_analysis():
    P_IR = 0.5
    P_RS = 0.5
    size = 50
    
    os.makedirs('fig', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Pick 20 values for the infection probability (P_{SI}) at regular intervals between 0.2 and 0.5.
    # As seen in the heatmap, this range contains the critical point that determines whether the epidemic will break out or die out.
    p_si_values = np.linspace(0.2, 0.5, 20)
    variances = []
    errors = []

    print(f"Starting Jacknife Variance Analysis (P_RS={P_RS}, size={size})")

    for p1 in p_si_values:
        # Calculate the Jacknife variance and error for each P_{SI} (infection probability).
        v, err = get_variance_jackknife(p1, P_IR, P_RS, size=size)
        variances.append(v)
        errors.append(err)
        print(f"p_SI: {p1:.3f} | Variance: {v:.5f} ± {err:.5f}")

    # Save data
    data_to_save = np.column_stack((p_si_values, variances, errors))
    np.savetxt('data/variance_jacknife_data.txt', data_to_save, 
               header="p_SI\tVariance\tJacknife_Error", delimiter='\t', fmt='%.6f')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(p_si_values, variances, yerr=errors, fmt='o-', 
                 color='crimson', ecolor='black', capsize=4, label='Jacknife Variance / N')
    
    plt.title(f'SIRS Susceptibility via Jacknife Resampling ($P_{{RS}}={P_RS}$)')
    plt.xlabel('$p_{S \\to I}$')
    plt.ylabel('Variance $\chi$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('fig/sirs_variance_jacknife.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_variance_analysis()