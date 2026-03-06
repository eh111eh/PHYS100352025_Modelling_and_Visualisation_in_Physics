import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

def update_sirs(grid, P_SI, P_IR, P_RS):
    S, I, R = 0, 1, 2
    L = grid.shape[0]
    new_grid = grid.copy()

    # Find neighbors using periodic boundary conditions (PBC)
    up = np.roll(grid, 1, axis=0)
    down = np.roll(grid, -1, axis=0)
    left = np.roll(grid, 1, axis=1)
    right = np.roll(grid, -1, axis=1)

    # Boolean mask: does a cell have at least one infected neighbor?
    has_infected_neighbour = (up == I) | (down == I) | (left == I) | (right == I)

    # Define transition masks based on probabilities
    # 1. S -> I: Must be S, have infected neighbor, and pass P_SI check
    mask_S_to_I = (grid == S) & has_infected_neighbour & (np.random.rand(L, L) < P_SI)
    
    # 2. I -> R: Must be I and pass P_IR check
    mask_I_to_R = (grid == I) & (np.random.rand(L, L) < P_IR)
    
    # 3. R -> S: Must be R and pass P_RS check
    mask_R_to_S = (grid == R) & (np.random.rand(L, L) < P_RS)

    # Apply updates
    new_grid[mask_S_to_I] = I
    new_grid[mask_I_to_R] = R
    new_grid[mask_R_to_S] = S

    return new_grid

def compare_scenarios():
    size = 100
    
    # Define scenarios with (P_SI, P_IR, P_RS)
    scenarios = [
        {"name": "Absorbing State", "p": (0.2, 0.5, 0.5)}, # Infection vanishes quickly, leaving everyone in the S (Susceptible) state.
        {"name": "Dynamic Equilibrium", "p": (0.5, 0.5, 0.5)}, # Dynamic equilibrium, where infection and recovery rates balance out.
        {"name": "Cyclic Waves", "p": (0.9, 0.1, 0.01)} # Cyclical infection, moving through the population like a wave.
    ]

    # Initialize grids for each scenario
    # S: 70%, I: 20%, R: 10% initially
    grids = [np.random.choice([0, 1, 2], size=(size, size), p=[0.7, 0.2, 0.1]) for _ in range(3)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    cmap = ListedColormap(['white', 'red', 'navy']) # S=White, I=Red, R=Navy
    
    ims = []
    for i, ax in enumerate(axes):
        im = ax.imshow(grids[i], cmap=cmap, vmin=0, vmax=2, origin='lower')
        ax.set_title(f"{scenarios[i]['name']}\n$P_{{SI}}, P_{{IR}}, P_{{RS}} = {scenarios[i]['p']}$")
        ax.axis('off')
        ims.append(im)

    # Add descriptive colorbar
    cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal', ticks=[0, 1, 2])
    cbar.ax.set_xticklabels(['S (Susceptible)', 'I (Infected)', 'R (Recovered)'])

    def animate(frame):
        nonlocal grids
        updated_ims = []
        for i in range(3):
            p_si, p_ir, p_rs = scenarios[i]['p']
            grids[i] = update_sirs(grids[i], p_si, p_ir, p_rs)
            ims[i].set_data(grids[i])
            updated_ims.append(ims[i])
        return updated_ims

    ani = animation.FuncAnimation(fig, animate, frames=300, interval=30, blit=True)
    
    # Adjust layout to prevent overlap with titles/colorbar
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95, wspace=0.2)
    plt.show()

if __name__ == "__main__":
    compare_scenarios()