import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from numba import njit

# ------- Directory Setup -------
base_dir = "checkpoint1"
figures_dir = os.path.join(base_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# ------- Parameters -------
L = 50                  
T = 1.5                 # Lower temperature shows phase separation better
beta = 1.0 / T
J = 1.0
steps_per_frame = L * L 
n_frames = 200

# ------- Initialise lattice -------
# Kawasaki conserves magnetization, so we start with a fixed ratio
rng = np.random.default_rng()
spins = rng.choice(np.array([-1, 1], dtype=np.int32), size=(L, L))

# ------- Numba Accelerated Kernels -------

@njit
def get_neighbours_sum(spins, x, y, L):
    return (
        spins[(x + 1) % L, y] +
        spins[(x - 1) % L, y] +
        spins[x, (y + 1) % L] +
        spins[x, (y - 1) % L]
    )

@njit
def kawasaki_sweep_fast(spins, L, beta, J):
    """Performs one full sweep of Kawasaki exchanges."""
    for _ in range(L * L):
        # Pick two random sites
        x1, y1 = np.random.randint(0, L), np.random.randint(0, L)
        x2, y2 = np.random.randint(0, L), np.random.randint(0, L)

        s1 = spins[x1, y1]
        s2 = spins[x2, y2]

        if s1 == s2:
            continue

        # Check if they are nearest neighbours
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        is_nn = (dx == 1 and dy == 0) or (dx == 0 and dy == 1) or \
                (dx == L-1 and dy == 0) or (dx == 0 and dy == L-1)

        if is_nn:
            # Correct for the shared bond
            dE = 2 * J * (s1 * (get_neighbours_sum(spins, x1, y1, L) - s2) + 
                          s2 * (get_neighbours_sum(spins, x2, y2, L) - s1))
        else:
            dE = 2 * J * (s1 * get_neighbours_sum(spins, x1, y1, L) + 
                          s2 * get_neighbours_sum(spins, x2, y2, L))

        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[x1, y1] = s2
            spins[x2, y2] = s1
            
    return spins

# ------- Animation functions -------

def update(frame):
    global spins
    spins = kawasaki_sweep_fast(spins, L, beta, J)
    im.set_data(spins)
    return [im]

fig, ax = plt.subplots()
im = ax.imshow(spins, cmap="coolwarm", vmin=-1, vmax=1, animated=True)
ax.set_title(f"Kawasaki Dynamics (T = {T})")
ax.set_xticks([])
ax.set_yticks([])

# ------- Run & Save Animation -------
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)

save_path = os.path.join(figures_dir, "kawasaki_animation.gif")
print(f"Saving animation to {save_path}...")

try:
    ani.save(save_path, writer='pillow', fps=20)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving: {e}")

plt.show()