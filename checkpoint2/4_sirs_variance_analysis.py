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

def get_variance_jackknife(p1, p2, p3, size=50, eq_sweeps=500, run_sweeps=10000, num_blocks=20):
    """
    Calculates variance and error bars using the Jackknife Resampling method.
    """
    grid = np.random.choice([0, 1, 2], size=(size, size))
    N_sites = size * size

    # 1. Equilibration
    for _ in range(eq_sweeps):
        grid = update_sirs(grid, p1, p2, p3)

    # 2. Measurement: Collect all raw data
    raw_infected = []
    for _ in range(run_sweeps):
        grid = update_sirs(grid, p1, p2, p3)
        raw_infected.append(np.count_nonzero(grid == 1))
    
    raw_infected = np.array(raw_infected)
    
    # 3. Binning: Divide data into blocks to reduce autocorrelation
    block_size = run_sweeps // num_blocks
    # Calculate means of I and I^2 for each block
    block_I = np.array([np.mean(raw_infected[i*block_size : (i+1)*block_size]) for i in range(num_blocks)])
    block_I2 = np.array([np.mean(raw_infected[i*block_size : (i+1)*block_size]**2) for i in range(num_blocks)])

    # 4. Jackknife Procedure
    # chi = (<I^2> - <I>^2) / N_sites
    jk_variances = []
    for i in range(num_blocks):
        # exclude i-th block and calculate average of the rest
        sum_I = (np.sum(block_I) - block_I[i]) / (num_blocks - 1)
        sum_I2 = (np.sum(block_I2) - block_I2[i]) / (num_blocks - 1)
        
        var_jk = (sum_I2 - sum_I**2) / N_sites
        jk_variances.append(var_jk)
    
    jk_variances = np.array(jk_variances)
    
    # Final estimate: average of all blocks
    mean_I = np.mean(block_I)
    mean_I2 = np.mean(block_I2)
    final_variance = (mean_I2 - mean_I**2) / N_sites
    
    # Jackknife Error formula: sqrt((K-1)/K * sum((jk_i - mean_jk)^2))
    error = np.sqrt((num_blocks - 1) * np.var(jk_variances))
    
    return final_variance, error

def run_variance_analysis():
    P_IR = 0.5
    P_RS = 0.5
    size = 50
    
    os.makedirs('fig', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    p_si_values = np.linspace(0.2, 0.5, 20)
    variances = []
    errors = []

    print(f"Starting Jacknife Variance Analysis (P_RS={P_RS}, size={size})")

    for p1 in p_si_values:
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