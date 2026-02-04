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
T = 2.27             # Set near critical temperature for interesting visuals
beta = 1.0 / T
J = 1.0             
steps_per_frame = L * L  
n_frames = 200

# ------- Initialise lattice -------
rng = np.random.default_rng()
spins = rng.choice(np.array([-1, 1], dtype=np.int32), size=(L, L))

# ------- Numba Accelerated Functions -------
@njit
def glauber_step_fast(spins, L, beta, J):
    """Performs one full Monte Carlo sweep (L*L steps) using Numba."""
    for _ in range(L * L):
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        
        # Periodic boundaries
        nb_sum = (spins[(x + 1) % L, y] + spins[(x - 1) % L, y] + 
                  spins[x, (y + 1) % L] + spins[x, (y - 1) % L])
        
        dE = 2.0 * J * spins[x, y] * nb_sum

        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[x, y] *= -1
    return spins

# ------- Animation function -------
def update(frame):
    global spins
    # Perform one sweep per frame
    spins = glauber_step_fast(spins, L, beta, J)
    im.set_data(spins)
    return [im]

# ------- Plot setup -------
fig, ax = plt.subplots()
im = ax.imshow(
    spins,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    animated=True
)

ax.set_title(f"Glauber Dynamics (T = {T})")
ax.set_xticks([])
ax.set_yticks([])

# ------- Run & Save Animation -------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=50,
    blit=True
)

# Save the animation
# Note: Requires 'pillow' for .gif or 'ffmpeg' for .mp4
save_path = os.path.join(figures_dir, "glauber_animation.gif")
print(f"Saving animation to {save_path}...")

try:
    ani.save(save_path, writer='pillow', fps=20)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Make sure you have 'pillow' installed (pip install pillow).")

plt.show()