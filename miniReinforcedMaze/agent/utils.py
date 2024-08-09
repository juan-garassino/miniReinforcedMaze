from typing import Tuple

def estimate_q_table_size(maze_size, num_actions):
    """
    Estimate the size of the Q-table for a maze environment.
    
    Args:
    maze_size (tuple): The dimensions of the maze (width, height).
    num_actions (int): The number of possible actions.
    
    Returns:
    int: Estimated number of entries in the Q-table.
    """
    width, height = maze_size
    num_positions = width * height
    
    # Assuming the state includes position and some local information (e.g., walls in 4 directions)
    num_local_states = 2**4  # 2 possibilities (wall or no wall) in 4 directions
    
    total_states = num_positions * num_local_states
    return total_states * num_actions

def calculate_reward(current_position, treasure_position, base_reward):
    distance = manhattan_distance(current_position, treasure_position)
    return base_reward + (1 / (distance + 1))  # Avoid division by zero

def manhattan_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> int:
    """
    Calculate the Manhattan distance between two points.

    Args:
    point1 (Tuple[int, int]): The (x, y) coordinates of the first point.
    point2 (Tuple[int, int]): The (x, y) coordinates of the second point.

    Returns:
    int: The Manhattan distance between the two points.
    """
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def calculate_reward(current_position: Tuple[int, int], treasure_position: Tuple[int, int], base_reward: float) -> float:
    """
    Calculate the reward based on the current position and the treasure position.

    Args:
    current_position (Tuple[int, int]): The current (x, y) position of the agent.
    treasure_position (Tuple[int, int]): The (x, y) position of the treasure.
    base_reward (float): The base reward for taking an action.

    Returns:
    float: The calculated reward.
    """
    distance = manhattan_distance(current_position, treasure_position)
    return base_reward + (1 / (distance + 1))  # Avoid division by zero


# Example usage
maze_size = (10, 10)
num_actions = 4  # Up, Down, Left, Right

estimated_size = estimate_q_table_size(maze_size, num_actions)
print(f"Estimated Q-table size: {estimated_size} entries")
print(f"Estimated memory usage (assuming 8 bytes per float): {estimated_size * 8 / (1024*1024):.2f} MB")