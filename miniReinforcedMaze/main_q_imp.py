from miniReinforcedMaze.generate.generate_maze import MazeEnvironment
from miniReinforcedMaze.agent.players import ImprovedQPlayer, train_improved_q_player, test_agent
from miniReinforcedMaze.params import *
from miniReinforcedMaze.agent.simulate import simulate_maze_game, simulate_maze_improve_game
from miniReinforcedMaze.preprocessing.cropping import crop
import os

def main():
    # Create maze
    maze_size = 16
    maze_density = 0.1
    maze_env = MazeEnvironment(maze_size, maze_size, density=maze_density)
    maze = maze_env.maze

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
    
    # Create Q-player
    action_size = len(POSSIBLE_ACTIONS)
    q_player = ImprovedQPlayer(action_size=action_size)

    trained_player = train_improved_q_player(
        player=q_player,
        maze_size=(maze_size, maze_size),
        maze_density=maze_density,
        episodes=100,
        max_steps=30,
        batch_size=32,
        show_every=1,
        possible_actions=POSSIBLE_ACTIONS,
        converted_directions=CONVERT_DIRECTIONS,
        codes=codes
    )

    print("Training completed.")

    # Simulate a game with the trained player
    print("\nSimulating a game with the trained player:")
    observations, result = simulate_maze_improve_game(
        maze=maze,
        player=trained_player,
        initial_position=maze_env.initial_position,
        possible_actions=POSSIBLE_ACTIONS,
        converted_directions=CONVERT_DIRECTIONS,
        codes=codes,
        verbose=True,
        show_gif=True,
        cropped_images=cropped_images
    )

    print(f"\nGame result: {result}")

if __name__ == "__main__":
    main()