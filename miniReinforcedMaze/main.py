import argparse
import numpy as np
from tqdm import tqdm
from miniReinforcedMaze.environment.generate_maze import MazeEnvironment, RenderMaze
from miniReinforcedMaze.agent.players import QPlayer, ImprovedQPlayer, PPOPlayer
from miniReinforcedMaze.agent.training import train_q_player, train_improved_q_player, train_ppo_player
#from miniReinforcedMaze.visualization.generate_animation import GifMaze

def parse_arguments():
    parser = argparse.ArgumentParser(description="AI Maze Explorer")
    
    # Maze generation arguments
    parser.add_argument("--width", type=int, default=10, help="Width of the maze")
    parser.add_argument("--height", type=int, default=10, help="Height of the maze")
    parser.add_argument("--density", type=float, default=0.5, help="Difficulty of the maze")
    
    # Player and training arguments
    parser.add_argument("--player", choices=['q', 'dqn', 'ppo'], default='q', help="Type of player to use")
    parser.add_argument("--episodes-per-maze", type=int, default=1000, help="Number of training episodes per maze")
    parser.add_argument("--max-mazes", type=int, default=100, help="Maximum number of mazes to train on")
    parser.add_argument("--max-steps", type=int, default=150, help="Maximum steps per episode")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for the player")
    parser.add_argument("--discount-factor", type=float, default=0.95, help="Discount factor for future rewards")
    parser.add_argument("--exploration-rate", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--exploration-decay", type=float, default=0.995, help="Exploration rate decay")
    parser.add_argument("--success-rate-threshold", type=float, default=0.95, help="Success rate threshold for early stopping")
    parser.add_argument("--evaluation-mazes", type=int, default=5, help="Number of mazes to use for evaluation")
    
    # Visualization arguments
    parser.add_argument("--show-training", action="store_true", help="Show training progress")
    parser.add_argument("--create-gif", action="store_true", help="Create a GIF of the trained agent's performance")
    parser.add_argument("--gif-duration", type=int, default=60, help="Duration of each frame in the GIF (in milliseconds)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output during training")
    parser.add_argument("--image-path", type=str, default=None, help="Path to the directory containing the sprite sheet")
    
    return parser.parse_args()

def evaluate_player(player, env, num_episodes, max_steps):
    successes = 0
    total_rewards = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        for _ in range(max_steps):
            action = player.move(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        total_rewards += episode_reward
        if done and reward > 0:  # Assuming positive reward means finding the treasure
            successes += 1
    success_rate = successes / num_episodes
    avg_reward = total_rewards / num_episodes
    return success_rate, avg_reward

def main():
    args = parse_arguments()
    
    print("🌟 Welcome to the AI Maze Explorer project! 🌟")

    # Create an initial environment to get state and action sizes
    env = MazeEnvironment(width=args.width, height=args.height, density=args.density)
    state_size = len(env.get_state())
    action_size = len(env.get_possible_actions())

    # Initialize the player
    if args.player == 'q':
        print("\n🧠 Initializing Q-Learning player...")
        player = QPlayer(state_size, action_size, learning_rate=args.learning_rate,
                         discount_factor=args.discount_factor, exploration_rate=args.exploration_rate,
                         exploration_decay=args.exploration_decay)
        train_func = train_q_player
    elif args.player == 'dqn':
        print("\n🤖 Initializing Deep Q-Network player...")
        player = ImprovedQPlayer(state_size=state_size, action_size=action_size,
                                learning_rate=args.learning_rate,
                                discount_factor=args.discount_factor,
                                exploration_rate=args.exploration_rate,
                                exploration_decay=args.exploration_decay)
        train_func = train_improved_q_player
    elif args.player == 'ppo':
        print("\n🚀 Initializing PPO player...")
        player = PPOPlayer(state_size, action_size, learning_rate=args.learning_rate,
                           gamma=args.discount_factor)
        train_func = train_ppo_player

    # Train the player on multiple mazes
    print(f"\n🏋️ Starting training on multiple mazes...")
    maze_count = 0
    while maze_count < args.max_mazes:
        print(f"\n🌟 Training on maze {maze_count + 1}")
        env = MazeEnvironment(width=args.width, height=args.height, density=args.density)
        print(f"🏗️ Created a {args.width}x{args.height} maze")
        
        player = train_func(env, player, num_episodes=args.episodes_per_maze, max_steps=args.max_steps,
                            show_training=args.show_training, verbose=args.verbose)
        
        maze_count += 1
        
        # Evaluate the player on multiple mazes
        print("\n📊 Evaluating player performance...")
        success_rates = []
        avg_rewards = []
        for eval_maze in range(args.evaluation_mazes):
            eval_env = MazeEnvironment(width=args.width, height=args.height, density=args.density)
            success_rate, avg_reward = evaluate_player(player, eval_env, num_episodes=10, max_steps=args.max_steps)
            success_rates.append(success_rate)
            avg_rewards.append(avg_reward)
            print(f"Evaluation Maze {eval_maze + 1}: Success Rate: {success_rate:.2f}, Average Reward: {avg_reward:.2f}")
        
        mean_success_rate = np.mean(success_rates)
        mean_avg_reward = np.mean(avg_rewards)
        print(f"Mean Success Rate: {mean_success_rate:.2f}")
        print(f"Mean Average Reward: {mean_avg_reward:.2f}")
        
        if mean_success_rate >= args.success_rate_threshold:
            print(f"\n🎉 Success rate threshold reached! Stopping training.")
            break
    
    if maze_count >= args.max_mazes:
        print(f"\n⚠️ Maximum number of mazes ({args.max_mazes}) reached. Stopping training.")

    # Test the trained player and create GIF if requested
    print("\n🧪 Testing the trained player on a new maze...")
    test_env = MazeEnvironment(width=args.width, height=args.height, density=args.density)

    print("\nCropping images...")

    width, height = 10, 10
    density = 0.2
    input_path = '/Users/juan-garassino/Code/juan-garassino/miniNetworks/miniReinforcedMaze/data'
    image_names = ["cave.png", "objects.png", "agus.png"]
    imagen_indexes = [16 * 8 + 2, 16 * 9, 16 * 20, 378, -1]
    num_steps = 20
    sprite_size = 32
    gif_duration = 500

    cropped_images, _ = RenderMaze.crop_images(
        input_path=input_path,
        image_names=image_names,
        # height=crop_height,
        # width=crop_width,
        sprite_size=sprite_size,
        return_indexes=imagen_indexes
    )

    gif_maze = RenderMaze(cropped_images=cropped_images) if args.create_gif else None
    test_episodes = 10
    for episode in range(test_episodes):
        state = test_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        if gif_maze is not None:
            gif_maze.add(maze=test_env.maze, position=test_env.current_pos)#, cropped_images=cropped_images)
        while not done and step_count < args.max_steps:
            action = player.move(state)
            next_state, reward, done, _ = test_env.step(action)
            total_reward += reward
            state = next_state
            step_count += 1
            if gif_maze is not None:
                gif_maze.add(maze=test_env.maze, position=test_env.current_pos)#, cropped_images=cropped_images)
        print(f"Test Episode {episode + 1}: Total Reward: {total_reward:.2f}, Steps: {step_count}")

    if gif_maze is not None:
        gif_maze.show(possible_actions=test_env.get_possible_actions(), gif_duration=args.gif_duration)
        print("\n🎬 GIF of the trained agent's performance has been created.")

    print("\n🎉 Project completed successfully!")

if __name__ == "__main__":
    main()