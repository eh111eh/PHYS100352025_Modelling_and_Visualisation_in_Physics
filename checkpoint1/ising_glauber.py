import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ------- Parameters -------
L = 50              # Lattice size (L x L)
T = 10.0             # Thermal energy k_B T
beta = 1.0 / T
J = 1.0             # Coupling constant
steps_per_frame = L * L  # One Monte Carlo sweep per frame

# ------- Initialise lattice -------
"""Initialise a 2D Ising lattice with random spins +-1."""
rng = np.random.default_rng()
spins = rng.choice([-1, 1], size=(L, L))

# ------- Helper functions -------
"""
I made the system using Glauber dynamics, where at each step a random spin
is selected and flipped according to the acceptance rule based on the local energy change.
The energy change depends only on the four nearest neighbours
"""
def neighbours_sum(x, y):
    """Sum of nearest neighbours with periodic boundary conditions."""
    return (
        spins[(x + 1) % L, y] +
        spins[(x - 1) % L, y] +
        spins[x, (y + 1) % L] +
        spins[x, (y - 1) % L]
    )

def deltaE_glauber(x, y):
    """Energy change for flipping one spin."""
    return 2 * J * spins[x, y] * neighbours_sum(x, y)

def glauber_step():
    """Single Glauber update."""
    x, y = rng.integers(0, L, size=2)
    dE = deltaE_glauber(x, y)

    if dE <= 0:
        spins[x, y] *= -1
    else:
        if rng.random() < np.exp(-beta * dE):
            spins[x, y] *= -1

# ------- Animation function -------
def update(frame):
    """
    Perform several Glauber steps per animation frame.
    Each animation frame corresponds to one Monte Carlo sweep, after which the spin configuration is updated on the screen.
    """
    for _ in range(steps_per_frame):
        glauber_step()

    im.set_data(spins)
    return [im]

# ------- Plot setup -------
"""
I visualise the lattice using a colour map where red and blue correspond to spin up and spin down.
The title indicates the dynamics and temperature, and I remove axis ticks for clarity.
"""
fig, ax = plt.subplots()
im = ax.imshow(
    spins,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    animated=True
)

ax.set_title(f"Glauber Dynamics Ising Model (T = {T})")
ax.set_xticks([])
ax.set_yticks([])

# ------- Run animation -------
"""
Finally, I run the animation to visualise the time evolution of the Kawasaki dynamics
"""
ani = animation.FuncAnimation(
    fig,
    update,
    frames=200,
    interval=50,
    blit=True
)

plt.show()
