import numpy as np
import os


def generate_maze(m, n, density=0.2):
    """
    Generate a maze of size m x n with randomly positioned holes and a treasure.
    The density parameter determines the probability of a cell being a hole.
    Returns the generated maze as a numpy array.
    """
    # Generate a random sample of CODE_LAND and CODE_HOLE based on the density
    maze = np.random.choice(
        a=[int(os.environ.get("CODE_LAND")), int(os.environ.get("CODE_HOLE"))],
        size=m * n,
        p=[1.0 - density, density],
    ).reshape((m, n))

    # Position the treasure at the bottom-right corner of the maze
    maze[m - 2, n - 2] = int(os.environ.get("CODE_TREASURE"))

    # Set the initial position to be land
    maze[int(os.environ.get("CODE_HOLE"))] = int(os.environ.get("CODE_LAND"))

    # Create hole boundaries on the edge of the maze
    for i in range(m):
        maze[i, 0] = int(os.environ.get("CODE_HOLE"))
        maze[i, n - 1] = int(os.environ.get("CODE_HOLE"))

    for j in range(n):
        maze[0, j] = int(os.environ.get("CODE_HOLE"))
        maze[m - 1, j] = int(os.environ.get("CODE_HOLE"))

    # Randomly flip the maze horizontally and vertically
    flip_horizontal = np.random.choice([-1, 1])
    flip_vertical = np.random.choice([-1, 1])

    return maze[::flip_horizontal, ::flip_vertical]  # Flip the maze and return
