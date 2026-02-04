import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse

# ------- Command-line arguments -------
parser = argparse.ArgumentParser(
    description="2D Ising model with Glauber or Kawasaki dynamics"
)

parser.add_argument(
    "--L",
    type=int,
    default=50,
    help="Linear system size (default: 50)"
)

parser.add_argument(
    "--T",
    type=float,
    default=2.0,
    help="Thermal energy k_B T (default: 2.0)"
)

parser.add_argument(
    "--dynamics",
    type=str,
    choices=["glauber", "kawasaki"],
    default="glauber",
    help="Dynamics type: glauber or kawasaki"
)

args = parser.parse_args()

# ------- Parameters -------
L = args.L
T = args.T
beta = 1.0 / T
J = 1.0
steps_per_frame = L * L

# ------- Initialise lattice -------
rng = np.random.default_rng()
spins = rng.choice([-1, 1], size=(L, L))

# Nearest-neighbour sum
def neighbours_sum(x, y):
    """
    Sum of nearest neighbours with periodic boundary conditions.
    """
    return (
        spins[(x + 1) % L, y] +
        spins[(x - 1) % L, y] +
        spins[x, (y + 1) % L] +
        spins[x, (y - 1) % L]
    )

# ------- Glauber dynamics -------
def deltaE_glauber(x, y):
    """
    Energy change for flipping a single spin.
    """
    return 2 * J * spins[x, y] * neighbours_sum(x, y)

def glauber_step():
    """
    Single Glauber Monte Carlo update.
    """
    x, y = rng.integers(0, L, size=2)
    dE = deltaE_glauber(x, y)

    if dE <= 0:
        spins[x, y] *= -1
    else:
        if rng.random() < np.exp(-beta * dE):
            spins[x, y] *= -1

# ------- Kawasaki dynamics -------
def deltaE_kawasaki(x1, y1, x2, y2):
    """
    Energy change for exchanging two spins using the simultaneous-exchange method.
    """
    if x1 == x2 and y1 == y2:
        return 0.0

    S1 = spins[x1, y1]
    S2 = spins[x2, y2]

    if S1 == S2:
        return 0.0

    nn = (
        (abs(x1 - x2) == 1 and y1 == y2) or
        (abs(y1 - y2) == 1 and x1 == x2) or
        (abs(x1 - x2) == L - 1 and y1 == y2) or
        (abs(y1 - y2) == L - 1 and x1 == x2)
    )

    if nn:
        n1 = neighbours_sum(x1, y1) - S2
        n2 = neighbours_sum(x2, y2) - S1
    else:
        n1 = neighbours_sum(x1, y1)
        n2 = neighbours_sum(x2, y2)

    return 2 * J * (S1 * n1 + S2 * n2)

def kawasaki_step():
    """
    Single Kawasaki Monte Carlo update.
    """
    x1, y1 = rng.integers(0, L, size=2)
    x2, y2 = rng.integers(0, L, size=2)

    dE = deltaE_kawasaki(x1, y1, x2, y2)

    if dE <= 0:
        spins[x1, y1], spins[x2, y2] = spins[x2, y2], spins[x1, y1]
    else:
        if rng.random() < np.exp(-beta * dE):
            spins[x1, y1], spins[x2, y2] = spins[x2, y2], spins[x1, y1]

# ------- Animation update -------
def update(frame):
    """
    One animation frame corresponds to one Monte Carlo sweep.
    """
    for _ in range(steps_per_frame):
        if args.dynamics == "glauber":
            glauber_step()
        else:
            kawasaki_step()

    im.set_data(spins)
    return [im]

# ------- Plot and animation -------
fig, ax = plt.subplots()
im = ax.imshow(
    spins,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    animated=True
)

ax.set_title(
    f"Ising Model ({args.dynamics.capitalize()} dynamics, T = {T})"
)
ax.set_xticks([])
ax.set_yticks([])

ani = animation.FuncAnimation(
    fig,
    update,
    frames=200,
    interval=50,
    blit=True
)

plt.show()
