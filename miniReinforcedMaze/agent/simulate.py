from miniReinforcedMaze.agent.actions import max_visibility, choose_action, apply_action
from miniReinforcedMaze.visualization.generate_animation import (
    GifMaze,
    show_maze,
)
from collections import defaultdict
import random
import numpy as np

def print_verbose_output(
    *,
    turn_counter=0,
    position=None,
    old_state=None,
    action=None,
    converted_directions=None,
    player=None,
    possible_actions=None,
    new_position=None,
    new_state=None,
    reward=0.0,
    maze=None,
    result=None,
    observations=None
):
    """
    Print verbose output for the maze game simulation.

    Args:
        turn_counter (int, optional): Current turn number. Defaults to 0.
        position (tuple, optional): Current position in the maze. Defaults to None.
        old_state (tuple, optional): State before the action. Defaults to None.
        action: Chosen action. Defaults to None.
        converted_directions (dict, optional): Mapping of actions to directions. Defaults to None.
        player: The player object. Defaults to None.
        possible_actions (dict, optional): Possible actions the player can take. Defaults to None.
        new_position (tuple, optional): New position after the action. Defaults to None.
        new_state (tuple, optional): State after the action. Defaults to None.
        reward (float, optional): Reward for the current step. Defaults to 0.0.
        maze (numpy.array, optional): The maze layout. Defaults to None.
        result (str, optional): Game result ('win' or 'lose'). Defaults to None.
        observations (list, optional): List of all observations. Defaults to None.
    """
    print("\n" + "="*40)
    print(f"Turn {turn_counter:3d}")
    print("="*40)

    print(f"Current Position: {position}")
    print(f"Old State: {list(old_state)}")

    print(f"Chosen Action: {converted_directions[action]}")
    if hasattr(player, "Q"):
        q_value = choose_action(Q_table=player.Q, possible_actions=possible_actions, state=old_state)[1]
        print(f"Q-Value for this action: {q_value:.4f}")

    print(f"New Position: {new_position}")
    print(f"New State: {list(new_state)}")
    print(f"Reward for this step: {reward:.2f}")
    
    # show_maze(maze=maze, position=new_position)

    if result:
        print("\n" + "="*40)
        print(f"Game Over: Player {'found the treasure' if result == 'win' else 'fell into a hole'}!")
        print(f"Final Reward: {reward:.2f}")
        if observations:
            print(f"Total Moves: {len(observations)}")
            if hasattr(player, 'i'):
                print(f"Exploited Moves: {player.i}")
                print(f"Exploitation Rate: {player.i / len(observations):.2f}")
        print("="*40 + "\n")

def simulate_maze_game(
    maze=None,
    player=None,
    initial_position=(0, 0),
    possible_actions=None,
    converted_directions=None,
    reward_step=0.0,
    reward_repeat=-0.1,
    reward_lose=-1.0,
    reward_win=1.0,
    codes=None,
    cropped_images=None,
    verbose=False,
    show_gif=False,
):
    if maze is None or player is None or possible_actions is None or converted_directions is None or codes is None:
        raise ValueError("maze, player, possible_actions, converted_directions, and codes must be provided")

    position = initial_position
    observations = []
    reward = 0.0
    result = None
    memory = defaultdict(lambda: 0)
    memory[initial_position] = 1

    if show_gif:
        gif_maze = GifMaze()
        gif_maze.add(maze=maze.copy(), position=position, cropped_images=cropped_images)

    turn_counter = 0

    while True:
        turn_counter += 1
        reward = reward_step

        old_state = (
            position
            + max_visibility(maze=maze, position=position, max_distance=5)
            + tuple(
                [
                    memory[apply_action(position=position, maze=maze, possible_actions=possible_actions, action=action)]
                    for _, action in possible_actions.items()
                ]
            )
        )

        if show_gif:
            gif_maze.add(maze=maze.copy(), position=position, cropped_images=cropped_images)

        state_array = np.array(old_state).reshape(1, -1)
        action = player.move(state=state_array)

        try:
            new_position = apply_action(position=position, maze=maze, possible_actions=possible_actions, action=action)
            print(f"New position after apply_action: {new_position}")
        except Exception as e:
            print(f"Error in apply_action: {str(e)}")
            print(f"player: {player}")
            print(f"player.Q: {player.Q if hasattr(player, 'Q') else 'No Q-table'}")
            raise

        if memory[new_position] == 1:
            reward = reward_repeat

        memory[new_position] = 1

        new_state = (
            new_position
            + max_visibility(maze=maze, position=new_position, max_distance=5)
            + tuple(
                [
                    memory[apply_action(position=new_position, maze=maze, possible_actions=possible_actions, action=action)]
                    for _, action in possible_actions.items()
                ]
            )
        )

        print(f"Reward: {reward}")

        if maze[new_position] == codes["HOLE"]:
            reward = reward_lose
            result = "lose"
            observations.append((old_state, action, reward, new_state))
            print("Game over: Fell into a hole")
            break
        elif maze[new_position] == codes["TREASURE"]:
            reward = reward_win
            result = "win"
            observations.append((old_state, action, reward, new_state))
            print("Game over: Found the treasure")
            break

        observations.append((old_state, action, reward, new_state))
        
        if verbose:
            print_verbose_output(
                turn_counter=turn_counter,
                position=position,
                old_state=old_state,
                action=action,
                converted_directions=converted_directions,
                player=player,
                possible_actions=possible_actions,
                new_position=new_position,
                new_state=new_state,
                reward=reward,
                maze=maze
            )

        position = new_position

    if verbose:
        print_verbose_output(
            turn_counter=turn_counter,
            position=position,
            old_state=old_state,
            action=action,
            converted_directions=converted_directions,
            player=player,
            possible_actions=possible_actions,
            new_position=new_position,
            new_state=new_state,
            reward=reward,
            maze=maze,
            result=result,
            observations=observations
        )

    if show_gif:
        gif_maze.add(maze=maze.copy(), position=position, cropped_images=cropped_images)
        gif_maze.show(possible_actions=possible_actions)

    return observations, result

def simulate_maze_improve_game(
    maze=None,
    player=None,
    initial_position=(0, 0),
    possible_actions=None,
    converted_directions=None,
    reward_step=0.0,
    reward_repeat=-0.1,
    reward_lose=-1.0,
    reward_win=1.0,
    codes=None,
    cropped_images=None,
    verbose=False,
    show_gif=False,
    expected_input_shape=(24,),  # New parameter for expected input shape
):
    if maze is None or player is None or possible_actions is None or converted_directions is None or codes is None:
        raise ValueError("maze, player, possible_actions, converted_directions, and codes must be provided")

    position = initial_position
    observations = []
    reward = 0.0
    result = None
    memory = defaultdict(lambda: 0)
    memory[initial_position] = 1

    if show_gif:
        gif_maze = GifMaze()
        gif_maze.add(maze=maze.copy(), position=position, cropped_images=cropped_images)

    turn_counter = 0
    first_step = True
    last_observation = None

    while True:
        turn_counter += 1
        reward = reward_step

        old_state = (
            position
            + max_visibility(maze=maze, position=position, max_distance=5)
            + tuple(
                [
                    memory[apply_action(position=position, maze=maze, possible_actions=possible_actions, action=action)]
                    for _, action in possible_actions.items()
                ]
            )
        )

        # Convert old_state to a numpy array and ensure it matches the expected input shape
        state_array = np.array(old_state).reshape(1, -1)

        # Trim or pad state_array to match the expected input shape
        if state_array.shape[1] > expected_input_shape[0]:
            state_array = state_array[:, :expected_input_shape[0]]
        elif state_array.shape[1] < expected_input_shape[0]:
            padding = expected_input_shape[0] - state_array.shape[1]
            state_array = np.pad(state_array, ((0, 0), (0, padding)), 'constant')

        action = player.move(state=state_array)

        try:
            new_position = apply_action(position=position, maze=maze, possible_actions=possible_actions, action=action)
            print(f"New position after apply_action: {new_position}")
        except Exception as e:
            print(f"Error in apply_action: {str(e)}")
            print(f"player: {player}")
            print(f"player.Q: {player.Q if hasattr(player, 'Q') else 'No Q-table'}")
            raise

        if memory[new_position] == 1:
            reward = reward_repeat

        memory[new_position] = 1

        new_state = (
            new_position
            + max_visibility(maze=maze, position=new_position, max_distance=5)
            + tuple(
                [
                    memory[apply_action(position=new_position, maze=maze, possible_actions=possible_actions, action=action)]
                    for _, action in possible_actions.items()
                ]
            )
        )

        print(f"Reward: {reward}")

        # Store the observation
        if not first_step:
            observations.append((old_state, action, reward, new_state))
        else:
            first_step = False

        # Update the last observation
        last_observation = (old_state, action, reward, new_state)

        # Check for game-ending conditions
        if maze[new_position] == codes["HOLE"]:
            reward = reward_lose
            result = "lose"
            if last_observation is not None:
                observations[-1] = last_observation  # Replace the last observation
            print("Game over: Fell into a hole")
            break
        elif maze[new_position] == codes["TREASURE"]:
            reward = reward_win
            result = "win"
            if last_observation is not None:
                observations[-1] = last_observation  # Replace the last observation
            print("Game over: Found the treasure")
            break

        position = new_position

        if verbose:
            print_verbose_output(
                turn_counter=turn_counter,
                position=position,
                old_state=old_state,
                action=action,
                converted_directions=converted_directions,
                player=player,
                possible_actions=possible_actions,
                new_position=new_position,
                new_state=new_state,
                reward=reward,
                maze=maze
            )

    if verbose:
        print_verbose_output(
            turn_counter=turn_counter,
            position=position,
            old_state=old_state,
            action=action,
            converted_directions=converted_directions,
            player=player,
            possible_actions=possible_actions,
            new_position=new_position,
            new_state=new_state,
            reward=reward,
            maze=maze,
            result=result,
            observations=observations
        )

    if show_gif:
        gif_maze.add(maze=maze.copy(), position=position, cropped_images=cropped_images)
        gif_maze.show(possible_actions=possible_actions)

    # Return the last observation and result
    return observations, result
