import random
import numpy as np

def choose_action(*, Q_table=None, possible_actions=None, state=None):
    print(f"choose_action called with state: {state}, possible_actions: {possible_actions}")
    if Q_table is None or possible_actions is None or state is None:
        raise ValueError("Q_table, possible_actions, and state must be provided")

    if not possible_actions:
        print("Warning: possible_actions is empty. Returning None.")
        return None, 0

    print(f"Q_table entries for this state: {[Q_table.get((action, state), 0) for action in possible_actions]}")

    taken_action = None
    current_value = float('-inf')
    for action in possible_actions:
        possible_value = Q_table.get((action, state), 0)
        print(f"Action: {action}, Q-value: {possible_value}")
        if possible_value > current_value:
            current_value = possible_value
            taken_action = action

    if taken_action is None:
        print("No best action found. Choosing a random action.")
        taken_action = random.choice(possible_actions)
        current_value = Q_table.get((taken_action, state), 0)

    print(f"choose_action returning: action={taken_action}, value={current_value}")
    return taken_action, current_value


def max_visibility(*, maze=None, position=None, max_distance=1):
    """
    Get the "vision" of the agent in the maze.

    Args:
    maze (numpy.array, optional): A 2D numpy array representing the maze. Defaults to None.
    position (tuple, optional): A tuple of (x, y) coordinates representing the agent's current position. Defaults to None.
    max_distance (int, optional): The maximum distance to check in each direction. Defaults to 1.

    Returns:
    tuple: A tuple containing the values of the cells in the agent's vision.
           1 represents a wall or out-of-bounds area.
    """
    if maze is None or position is None:
        raise ValueError("maze and position must be provided")

    vision = []
    for distance in range(1, max_distance + 1):
        vision += [
            maze[position[0], position[1] - distance] if position[1] - distance >= 0 else 1,
            maze[position[0], position[1] + distance] if position[1] + distance < maze.shape[1] else 1,
            maze[position[0] - distance, position[1]] if position[0] - distance >= 0 else 1,
            maze[position[0] + distance, position[1]] if position[0] + distance < maze.shape[0] else 1,
        ]
    return tuple(vision)


def apply_action(*, position=None, maze=None, possible_actions=None, action=None):
    """
    Apply the chosen action and update the agent's position in the maze.

    Args:
    position (tuple, optional): A tuple of (x, y) coordinates representing the agent's current position. Defaults to None.
    maze (numpy.array, optional): A 2D numpy array representing the maze. Defaults to None.
    possible_actions (dict, optional): A dictionary mapping action names to their corresponding values. Defaults to None.
    action: The action to be applied. Defaults to None.

    Returns:
    tuple: A tuple of (x, y) coordinates representing the agent's new position after applying the action.

    Raises:
    Exception: If the action is not recognized.
    ValueError: If any of the required arguments are not provided.
    """
    # print("apply_action called with:")
    # print(f"  position: {position} (type: {type(position)})")
    # print(f"  maze: shape={maze.shape if maze is not None else None} (type: {type(maze)})")
    # print(f"  possible_actions: {possible_actions} (type: {type(possible_actions)})")
    # print(f"  action: {action} (type: {type(action)})")

    if position is None or maze is None or possible_actions is None or action is None:
        raise ValueError(f"All arguments must be provided. Received: position={position}, maze_shape={maze.shape if maze is not None else None}, possible_actions={possible_actions}, action={action}")

    if action not in possible_actions.values():
        raise ValueError(f"Invalid action: {action}. Possible actions are: {possible_actions}")

    # print(f"Applying action: {action}")
    if action == possible_actions["UP"]:
        return (position[0], max(position[1] - 1, 0))
    elif action == possible_actions["DOWN"]:
        return (position[0], min(position[1] + 1, maze.shape[0] - 1))
    elif action == possible_actions["LEFT"]:
        return (max(position[0] - 1, 0), position[1])
    elif action == possible_actions["RIGHT"]:
        return (min(position[0] + 1, maze.shape[1] - 1), position[1])
    
    raise Exception(f"Action {action} not understood.")