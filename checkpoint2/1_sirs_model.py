import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

def update_sirs(grid, P_SI, P_IR, P_RS):
    S, I, R = 0, 1, 2
    L, _ = grid.shape 

    new_grid = grid.copy()

    up = np.roll(grid, 1, axis=0)
    down = np.roll(grid, -1, axis=0)
    left = np.roll(grid, 1, axis=1)
    right = np.roll(grid, -1, axis=1)

    has_infected_neighbour = (up == I) | (down == I) | (left == I) | (right == I)

    mask_S_to_I = (grid == S) & has_infected_neighbour & (np.random.rand(L, L) < P_SI)

    mask_I_to_R = (grid == I) & (np.random.rand(L, L) < P_IR)

    mask_R_to_S = (grid == R) & (np.random.rand(L, L) < P_RS)

    new_grid[mask_S_to_I] = I
    new_grid[mask_I_to_R] = R
    new_grid[mask_R_to_S] = S

    return new_grid

def main():
    parser = argparse.ArgumentParser(description="SIRS Model Simulation")
    parser.add_argument('-size', type=int, default=50)
    parser.add_argument('-P_SI', type=float, default=0.5)
    parser.add_argument('-P_IR', type=float, default=0.2)
    parser.add_argument('-P_RS', type=float, default=0.05)
    args = parser.parse_args()

    grid = np.random.choice([0, 1, 2], size=(args.size, args.size), p=[0.5, 0.2, 0.3])

    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = ListedColormap(['white', 'red', 'navy'])
    img = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, origin='lower')
    ax.set_title("SIRS Model Simulation")

    def animate(frame):
        nonlocal grid
        grid = update_sirs(grid, args.P_SI, args.P_IR, args.P_RS)
        img.set_data(grid)
        return [img]

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    main()