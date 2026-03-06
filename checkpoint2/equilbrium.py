import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
os.makedirs("fig", exist_ok=True)
os.makedirs("data", exist_ok=True)

N = 50
MAX_STEPS = 5000
NUM_SIMULATIONS = 1000

def update_grid(grid):
    neighbors = (
        np.roll(np.roll(grid, 1, 0), 1, 1) + np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) + np.roll(np.roll(grid, -1, 0), -1, 1) +
        np.roll(grid, 1, 0) + np.roll(grid, -1, 0) +
        np.roll(grid, 1, 1) + np.roll(grid, -1, 1)
    )
    new_grid = np.zeros((N, N), dtype=int)
    new_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
    new_grid[(grid == 0) & (neighbors == 3)] = 1
    return new_grid

def get_equilibrium_time(size):
    # Initial density 15%
    grid = np.random.choice([0, 1], size*size, p=[0.85, 0.15]).reshape(size, size)
    history_list = []
    history_set = set()
    
    for t in range(MAX_STEPS):
        grid_hash = grid.tobytes()
        
        # Check for static or oscillating equilibrium
        if grid_hash in history_set:
            return t
        
        history_list.append(grid_hash)
        history_set.add(grid_hash)
        if len(history_list) > 20:
            oldest = history_list.pop(0)
            history_set.remove(oldest)
            
        grid = update_grid(grid)
    return MAX_STEPS

def run_random_batch():
    print(f"Running {NUM_SIMULATIONS} trials...")
    equil_times = []
    for i in range(NUM_SIMULATIONS):
        equil_times.append(get_equilibrium_time(N))
        if (i+1) % 100 == 0:
            print(f"Progress: {i+1}/{NUM_SIMULATIONS}")

    np.savetxt("data/equilibrium_times.dat", equil_times, fmt="%d")
    
    # Filter for successful convergences to show the long-tail distribution
    valid_data = [t for t in equil_times if t < MAX_STEPS]
    
    plt.figure(figsize=(8, 5))
    plt.hist(valid_data, bins=80, color='forestgreen', edgecolor='black', alpha=0.7)
    plt.title("Equilibration Time Distribution (Random Initial Condition)")
    plt.xlabel("Steps to Equilibrate")
    plt.ylabel("Frequency (Count)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("fig/equilibrium_histogram.png")
    
    print(f"Average time: {np.mean(valid_data):.2f} steps")
    print(f"Unfinished (at MAX_STEPS): {equil_times.count(MAX_STEPS)}")

if __name__ == "__main__":
    run_random_batch()