from miniReinforcedMaze.preprocessing.cropping import crop
from miniReinforcedMaze.generate.generate_maze import MazeEnvironment
from miniReinforcedMaze.agent.players import RandomPlayer, QPlayer, train_q_player, ImprovedQPlayer, train_improved_q_player
from miniReinforcedMaze.agent.simulate import simulate_maze_game
from collections import defaultdict
from params import *
import os

# Get the directory of the current file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Join with another path (e.g., "subfolder/file.txt")
full_path = os.path.join(current_directory, "..", "data")

cropped_images, returned_indexes = crop(
    full_path,
    image_names,
    height=16,
    width=16,
    sprite_size=32,
    show=False,
    return_indexes=imagen_indexes,
)

maze = MazeEnvironment.generate_maze(30, 30, density=0.1)

random_player = RandomPlayer()  # Creates the player

# print(maze)

observations, result = simulate_maze_game(
    maze=maze,
    player=random_player,
    initial_position=INITIAL_POSITION,
    possible_actions=POSSIBLE_ACTIONS,
    converted_directions=CONVERT_DIRECTIONS,
    reward_step=REWARD_STEP,
    reward_repeat=REWARD_REPEAT,
    reward_lose=REWARD_LOSE,
    reward_win=REWARD_WIN,
    codes=codes,
    cropped_images=cropped_images,
    verbose=False,
    show_gif=False
)

# simulate_maze_game(maze, random_player, show_gif=True)  # with the respective maze and the respective player

# print(result)

q_player = QPlayer(defaultdict(lambda: 0), explore_factor=1.0)

# q_player = train_q_player(q_player, maze, cropped_images, amount_games=10000)

# Train the Q-player
q_player = train_q_player(
    q_player=q_player,
    maze=maze,
    cropped_images=cropped_images,
    num_mazes=1000,
    episodes_per_maze=50,
    initial_position=INITIAL_POSITION,
    possible_actions=POSSIBLE_ACTIONS,
    convert_directions=CONVERT_DIRECTIONS,
    reward_step=REWARD_STEP,
    reward_repeat=REWARD_REPEAT,
    reward_lose=REWARD_LOSE,
    reward_win=REWARD_WIN,
    codes=codes
)

# Create a new Q-player with the trained Q-table and no exploration
q_player = QPlayer(Q=q_player.Q, explore_factor=0.3)

# Simulate a maze game with the trained player
observations, result = simulate_maze_game(
    maze=maze,
    player=q_player,
    initial_position=INITIAL_POSITION,
    possible_actions=POSSIBLE_ACTIONS,
    converted_directions=CONVERT_DIRECTIONS,
    reward_step=REWARD_STEP,
    reward_repeat=REWARD_REPEAT,
    reward_lose=REWARD_LOSE,
    reward_win=REWARD_WIN,
    codes=codes,
    cropped_images=cropped_images,
    verbose=False,
    show_gif=True
)

# print(result)

improved_q_player = ImprovedQPlayer()
