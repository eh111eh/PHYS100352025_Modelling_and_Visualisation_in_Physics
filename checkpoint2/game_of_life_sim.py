import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N = 50  # Lattice size (50x50)
STEPS = 200 # Max steps for animation

def update_grid(grid):
    """Applies Conway's Game of Life rules with Periodic Boundary Conditions."""
    # Count 8 neighbors using np.roll for periodic boundaries 
    neighbors = (
        np.roll(np.roll(grid, 1, 0), 1, 1) + np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) + np.roll(np.roll(grid, -1, 0), -1, 1) +
        np.roll(grid, 1, 0) + np.roll(grid, -1, 0) +
        np.roll(grid, 1, 1) + np.roll(grid, -1, 1)
    )

    new_grid = np.copy(grid)
    # Rule 1 & 3: Death by underpopulation or overpopulation
    new_grid[(grid == 1) & ((neighbors < 2) | (neighbors > 3))] = 0
    # Rule 4: Birth in dead cell with exactly 3 neighbors
    new_grid[(grid == 0) & (neighbors == 3)] = 1
    # Rule 2: Survival with 2 or 3 neighbors (implicitly handled)
    
    return new_grid

def get_initial_state(mode, size):
    """Returns the initial lattice based on user choice."""
    grid = np.zeros((size, size))
    if mode == '1': # Random
        grid = np.random.choice([0, 1], size*size, p=[0.8, 0.2]).reshape(size, size)
    elif mode == '2': # Oscillator (Blinker)
        grid[25, 24:27] = 1
    elif mode == '3': # Glider (Spaceship)
        grid[2, 3] = 1
        grid[3, 4] = 1
        grid[4, 2:5] = 1
    return grid

def calculate_com(grid):
    """Calculates the center of mass of active sites."""
    active_sites = np.argwhere(grid == 1)
    if len(active_sites) == 0: return None
    return np.mean(active_sites, axis=0)

def animate(i, img, grid_ref, ax, title_base):
    grid_ref[0] = update_grid(grid_ref[0])
    img.set_data(grid_ref[0])
    ax.set_title(f"{title_base}")
    return img,

def run_simulation():
    print("Select Initial Condition:")
    print("1: Random\n2: Oscillator (Blinker)\n3: Glider (Spaceship)")
    choice = input("Choice: ")

    titles = {
        '1': "Game of Life: Random",
        '2': "Game of Life: Blinker (Oscillator)",
        '3': "Game of Life: Glider (Spaceship)"
    }
    selected_title = titles.get(choice, "Game of Life")

    grid = get_initial_state(choice, N)
    grid_ref = [grid] # Reference for animation update

    # Animation setup
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(grid, interpolation='nearest', cmap='binary', origin='lower')

    ax.set_title(selected_title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Lattice X-axis")
    ax.set_ylabel("Lattice Y-axis")
    ax.set_xticks(np.arange(0, N+1, 10))
    ax.set_yticks(np.arange(0, N+1, 10))

    ani = animation.FuncAnimation(fig, animate, fargs=(img, grid_ref, ax, selected_title),
                                  frames=STEPS, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    run_simulation()