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
    for _ in range(eq_sweeps):
        grid = update_sirs(grid, p1, p2, p3)
        
    # 2. Measurement
    counts = []
    for _ in range(run_sweeps):
        grid = update_sirs(grid, p1, p2, p3)
        counts.append(np.count_nonzero(grid == 1))
        
    return np.mean(counts) / (size * size)

def generate_heatmap():
    # Parameters
    size = 50
    p2 = 0.5  # Fixed P_IR
    resolution = 0.05
    p_values = np.arange(0, 1.0001, resolution)
    
    grid_size = len(p_values)
    heatmap_data = np.zeros((grid_size, grid_size))

    print(f"Starting Heatmap Generation ({grid_size}x{grid_size} points)...")

    for i, p1 in enumerate(p_values):      # x-axis: P_SI
        for j, p3 in enumerate(p_values):  # y-axis: P_RS
            heatmap_data[j, i] = get_avg_infection(p1, p2, p3, size=size)
        
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