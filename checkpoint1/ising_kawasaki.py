import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ------- Parameters -------
"""
Here I define the physical parameters of the Ising model and the numerical parameters
of the simulation. One Monte Carlo sweep corresponds to L squared attempted updates.”
"""
L = 50                  # Lattice size
T = 2.0                 # Thermal energy k_B T
beta = 1.0 / T
J = 1.0
steps_per_frame = L * L # One Monte Carlo sweep per frame

"""
I initialise the lattice with random spins +-1.
In Kawasaki dynamics, the total magnetisation is conserved,
so whatever magnetisation we start with remains fixed throughout the simulation.
"""
# ------- Initialise lattice -------
rng = np.random.default_rng()

# Random initial spins (magnetisation is fixed once initialised)
spins = rng.choice([-1, 1], size=(L, L))

# ------- Helper functions -------
def neighbours_sum(x, y):
    """
    Sum of nearest neighbours with periodic boundary conditions.
    This function computes the local field acting on a spin from its four nearest neighbours.
    Periodic boundary conditions are implemented using modulo arithmetic.
    """
    return (
        spins[(x + 1) % L, y] +
        spins[(x - 1) % L, y] +
        spins[x, (y + 1) % L] +
        spins[x, (y - 1) % L]
    )

def deltaE_kawasaki(x1, y1, x2, y2):
    """
    Energy change for exchanging spins at (x1,y1) and (x2,y2)
    Computed as a simultaneous exchange.

    This function computes the energy change associated with exchanging two spins.
    I use the simultaneous-exchange picture rather than two consecutive flips.
    If the two randomly chosen lattice sites are identical, the exchange does nothing and the energy change is zero.
    """
    if x1 == x2 and y1 == y2:
        return 0.0

    """I store the values of the two spins involved in the exchange for clarity."""
    S1 = spins[x1, y1]
    S2 = spins[x2, y2]

    # If spins are equal, exchange does nothing
    """If the two spins have the same value, exchanging them leaves the configuration unchanged, so the energy change is zero."""
    if S1 == S2:
        return 0.0

    # Check if nearest neighbours
    """Here I check whether the two spins are nearest neighbours, including across periodic boundaries. This is important because their mutual bond requires special treatment."""
    nn = (
        (abs(x1 - x2) == 1 and y1 == y2) or
        (abs(y1 - y2) == 1 and x1 == x2) or
        (abs(x1 - x2) == L - 1 and y1 == y2) or
        (abs(y1 - y2) == L - 1 and x1 == x2)
    )

    """
    If the spins are nearest neighbours, I subtract their mutual interaction. This is because exchanging two spins does not change their product, so their bond energy cancels out.
    If the spins are not neighbours, all their nearest-neighbour bonds contribute normally to the energy change.
    Based on the condition, it gives the total energy change from exchanging the two spins. Only bonds connecting each spin to the rest of the lattice contribute.
    """
    if nn:
        # Exclude each other from neighbour sums
        n1 = neighbours_sum(x1, y1) - S2
        n2 = neighbours_sum(x2, y2) - S1
    else:
        n1 = neighbours_sum(x1, y1)
        n2 = neighbours_sum(x2, y2)

    return 2 * J * (S1 * n1 + S2 * n2)

def kawasaki_step():
    """
    Single Kawasaki update.
    I randomly select two lattice sites where a possible spin exchange will be attempted,
    compute the energy change associated with exchanging the two spins.
    """
    x1, y1 = rng.integers(0, L, size=2)
    x2, y2 = rng.integers(0, L, size=2)

    dE = deltaE_kawasaki(x1, y1, x2, y2)

    """
    If the energy decreases or stays the same, the exchange is accepted unconditionally.
    If the energy increases, the exchange is accepted with Metropolis probability, ensuring detailed balance.
    """
    if dE <= 0:
        spins[x1, y1], spins[x2, y2] = spins[x2, y2], spins[x1, y1]
    else:
        if rng.random() < np.exp(-beta * dE):
            spins[x1, y1], spins[x2, y2] = spins[x2, y2], spins[x1, y1]

# ------- Animation update -------
def update(frame):
    """
    Each animation frame corresponds to one Monte Carlo sweep, after which the spin configuration is updated on the screen.
    """
    for _ in range(steps_per_frame):
        kawasaki_step()
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

ax.set_title(f"Kawasaki Dynamics Ising Model (T = {T})")
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
