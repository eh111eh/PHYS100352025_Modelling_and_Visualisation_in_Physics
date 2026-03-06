import numpy as np
import matplotlib.pyplot as plt

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

def get_avg_infection(p1, p2, p3, size=50, eq_sweeps=100, run_sweeps=500):
    """Runs a single simulation and returns the mean infection fraction."""
    # Initialize random grid
    grid = np.random.choice([0, 1, 2], size=(size, size))
    
    # 1. Equilibration
    # Run 100 iterations in advance until initial randomness fades and the system reaches a stable state.
    for _ in range(eq_sweeps):
        grid = update_sirs(grid, p1, p2, p3)
        
    # 2. Measurement
    counts = []
    for _ in range(run_sweeps):
        grid = update_sirs(grid, p1, p2, p3)
        # Count the number of infected individuals (1) in the current grid and store it in a list.
        counts.append(np.count_nonzero(grid == 1))

    # Divide the average number of infected individuals by the total number of grid cells (size x size) to return the average infection rate.   
    return np.mean(counts) / (size * size)

def generate_heatmap():
    """
    Run experiments across all combinations of infection probability (P_{SI})
    and immunity loss probability (P_{RS}).
    """
    # Parameters
    size = 50
    p2 = 0.5  # Fixed recovery probability P_IR
    resolution = 0.05 # Test from 0 to 1 in 0.05 increments (21 steps).
    p_values = np.arange(0, 1.0001, resolution) # Create a test list by dividing the probability into 5% increments from 0% to 100%.
    
    grid_size = len(p_values) # Number of probability test points.
    heatmap_data = np.zeros((grid_size, grid_size))

    print(f"Starting Heatmap Generation ({grid_size}x{grid_size} points)...")

    # Run nested loops to simulate all probability combinations.
    for i, p1 in enumerate(p_values):      # x-axis: P_SI
        for j, p3 in enumerate(p_values):  # y-axis: P_RS
            heatmap_data[j, i] = get_avg_infection(p1, p2, p3, size=size) # Save the average infection rate at each coordinate (j, i).
        
        print(f"Progress: {((i+1)/grid_size)*100:.1f}%")

    # Save data to file
    np.savetxt('data/heatmap_data.txt', heatmap_data, fmt='%.6f', delimiter='\t',
               header=f"SIRS Heatmap Data (p_IR={p2})\nRows: p_RS (0 to 1), Cols: p_SI (0 to 1)")
    print("\nNumerical data saved to 'data/heatmap_data.txt'")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, origin='lower', extent=[0, 1, 0, 1], 
               aspect='auto', cmap='magma')
    
    plt.colorbar(label='Average Infected Fraction $\\langle I \\rangle / N$')
    plt.xlabel('$p_{S \\to I}$ (Infection)')
    plt.ylabel('$p_{R \\to S}$ (Immunity Loss)')
    plt.title(f'SIRS Phase Diagram ($p_{{I \\to R}} = {p2}$)')
    
    plt.savefig('fig/sirs_heatmap.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_heatmap()