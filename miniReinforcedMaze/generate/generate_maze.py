import numpy as np
import os
from miniReinforcedMaze.agent.utils import calculate_reward

class MazeEnvironment:
    def __init__(self, m, n, density=0.2):
        self.m = m
        self.n = n
        self.maze = self.generate_maze(m, n, density)
        self.initial_position = self.find_initial_position()
        self.treasure_position = self.find_treasure_position()
        self.current_position = self.initial_position
        self.base_reward = -0.1

    def generate_maze(self, m, n, density=0.2):
        # Generate a random sample of CODE_LAND and CODE_HOLE based on the density
        maze = np.random.choice(
            a=[int(os.environ.get("CODE_LAND")), int(os.environ.get("CODE_HOLE"))],
            size=m * n,
            p=[1.0 - density, density],
        ).reshape((m, n))

        # Position the treasure at the bottom-right corner of the maze
        maze[m - 2, n - 2] = int(os.environ.get("CODE_TREASURE"))

        # Set the initial position to be land
        maze[1, 1] = int(os.environ.get("CODE_LAND"))

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

    def find_initial_position(self):
        return (1, 1)  # Assuming the initial position is always (1, 1)

    def find_treasure_position(self):
        return np.argwhere(self.maze == int(os.environ.get("CODE_TREASURE")))[0]

    def step(self, action):
        new_position = self.apply_action(self.current_position, action)
        
        if self.is_valid_position(new_position):
            self.current_position = new_position

        reward = self.get_reward()
        done = self.is_done()

        return self.current_position, reward, done

    def get_reward(self):
        if self.current_position == tuple(self.treasure_position):
            return 1.0  # Big positive reward for finding the treasure
        elif not self.is_valid_position(self.current_position):
            return -1.0  # Penalty for hitting a wall
        else:
            return calculate_reward(self.current_position, self.treasure_position, self.base_reward)

    def is_valid_position(self, position):
        x, y = position
        if 0 <= x < self.m and 0 <= y < self.n:
            return self.maze[x, y] != int(os.environ.get("CODE_HOLE"))
        return False

    def is_done(self):
        return tuple(self.current_position) == tuple(self.treasure_position)

    def apply_action(self, position, action):
        x, y = position
        if action == 0:  # Up
            return (x-1, y)
        elif action == 1:  # Down
            return (x+1, y)
        elif action == 2:  # Left
            return (x, y-1)
        elif action == 3:  # Right
            return (x, y+1)
        else:
            raise ValueError("Invalid action")

    def reset(self):
        self.current_position = self.initial_position
        return self.current_position
