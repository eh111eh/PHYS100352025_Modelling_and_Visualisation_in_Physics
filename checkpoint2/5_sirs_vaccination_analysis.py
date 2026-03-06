import numpy as np
import matplotlib.pyplot as plt

def update_sirs_with_immunity(grid, immune_mask, p1, p2, p3):
    """
    immune_mask: Boolean array where True means the site is permanently in R state.
    """
    S, I, R = 0, 1, 2
    L = grid.shape[0]
    new_grid = grid.copy()

    # PBC Neighbors
    up = np.roll(grid, 1, axis=0)
    down = np.roll(grid, -1, axis=0)
    left = np.roll(grid, 1, axis=1)
    right = np.roll(grid, -1, axis=1)

    has_infected_neighbour = (up == I) | (down == I) | (left == I) | (right == I)

    # Transition Masks
    mask_S_to_I = (grid == S) & has_infected_neighbour & (np.random.rand(L, L) < p1)
    mask_I_to_R = (grid == I) & (np.random.rand(L, L) < p2)
    # R -> S: Only happens for sites that are NOT permanently immune
    mask_R_to_S = (grid == R) & (~immune_mask) & (np.random.rand(L, L) < p3)

    new_grid[mask_S_to_I] = I
    new_grid[mask_I_to_R] = R
    new_grid[mask_R_to_S] = S # Permanent immune individuals are excluded and remain in the R

    return new_grid

def get_avg_infection_with_f_im(f_im, p1, p2, p3, size=50, eq_sweeps=100, run_sweeps=500):
    """
    Runs simulation for a specific fraction of permanently immune agents.
    """
    N = size * size
    grid = np.random.choice([0, 1, 2], size=(size, size))
    
    # Define permanent immunity mask
    # We randomly pick a fraction f_im of the total sites
    immune_mask = np.random.rand(size, size) < f_im
    
    # Set those sites to Recovered (R) initially
    grid[immune_mask] = 2

    # 1. Equilibration
    for _ in range(eq_sweeps):
        grid = update_sirs_with_immunity(grid, immune_mask, p1, p2, p3)

    # 2. Measurement
    infected_fractions = []
    for _ in range(run_sweeps):
        grid = update_sirs_with_immunity(grid, immune_mask, p1, p2, p3)
        # Store the infected population fraction in the infected_fractions list.
        infected_fractions.append(np.count_nonzero(grid == 1) / N)
        
    return np.mean(infected_fractions)

def run_vaccination_study():
    # Parameters from task: p_SI = p_IR = p_RS = 0.5
    p1 = p2 = p3 = 0.5
    size = 50
    
    # Immune fraction range from 0 (no immunity) to 1 (all immune): 21 levels
    f_im_values = np.linspace(0, 1.0, 21)
    avg_infections = []

    print(f"Starting Immunity Study (P_SI=P_IR=P_RS=0.5)...")

    # Open a file to save the numerical data
    with open('checkpoint2/data/vaccination_data.txt', 'w') as f_data:
        # Write a header for the datafile
        f_data.write("# f_Im\tAvg_Infected_Fraction\n")
        
        # Run the simulation for each vaccination rate (f) to calculate the average infection fraction.
        for f in f_im_values:
            mean_i = get_avg_infection_with_f_im(f, p1, p2, p3, size=size)
            avg_infections.append(mean_i)
            
            # Save to file
            f_data.write(f"{f:.4f}\t{mean_i:.6f}\n")
            print(f"Immune Fraction f_Im: {f:.2f} | Avg Infected Fraction: {mean_i:.4f}")

    print("\nNumerical data saved to 'data/vaccination_data.txt'")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(f_im_values, avg_infections, 'bo-', markersize=6, linewidth=2)
    
    plt.title(f'Effect of Permanent Immunity ($P_{{SI}}=P_{{IR}}=P_{{RS}}=0.5$)')
    plt.xlabel('Immune Fraction $f_{Im}$')
    plt.ylabel('Average Infected Fraction $\\langle I \\rangle / N$')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Optional: Mark the point where infection is prevented (close to 0)
    # plt.axhline(0, color='red', linestyle='-', alpha=0.3)
    
    plt.savefig('checkpoint2/fig/sirs_vaccination_study.png', dpi=300)
    print("\nPlot saved as 'sirs_vaccination_study.png'")
    plt.show()

if __name__ == "__main__":
    run_vaccination_study()