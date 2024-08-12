import numpy as np
from tqdm import tqdm
from miniReinforcedMaze.environment.generate_maze import MazeEnvironment
from miniReinforcedMaze.agent.simulate import simulate_maze_game

def train_q_player(env, player, num_episodes=1000, max_steps=500, show_training=False, verbose=False):
    print("🧠 Welcome to Q-Learning Academy! 🎓")
    print(f"Our eager student is about to embark on a {num_episodes}-episode journey of learning and exploration.")
    
    print("📚 Let the learning begin!")
    for episode in tqdm(range(num_episodes), desc="Training Progress", disable=not show_training):
        env.reset()
        observations, result = simulate_maze_game(env.maze, player, max_steps=max_steps)
        
        if verbose and episode % 100 == 0:
            print(f"\n📊 Episode {episode}: Our student {'found the treasure' if result == 'win' else 'is still learning'}.")
            print(f"   Current exploration rate: {player.epsilon:.2f}")
            print(f"   Q-table size: {len(player.Q)} entries")
    
    if verbose:
        print("\n🎉 Graduation Day! 🎓")
        print(f"Our Q-Learning student has completed {num_episodes} episodes of training.")
        print(f"Final Q-table size: {len(player.Q)} entries")
        print("Time to test our graduate in the real world!")
    
    return player

from tqdm import tqdm
from miniReinforcedMaze.agent.simulate import simulate_maze_game

from tqdm import tqdm

def train_improved_q_player(env, player, num_episodes=1000, max_steps=500, show_training=False, verbose=False):
    print("🤖 Welcome to the Deep Q-Network Dojo! 🥋")
    print(f"Our AI apprentice is ready for {num_episodes} intense training sessions.")
    
    # Ensure we have all necessary parameters from the environment
    possible_actions = env.get_possible_actions()
    converted_directions = env.get_converted_directions()
    codes = env.get_codes()
    
    print("🏋️ Let the neural training commence!")
    for episode in tqdm(range(num_episodes), desc="Training Progress", disable=not show_training):
        env.reset()
        initial_state = env.get_state()
        
        # Check if state size has changed
        if len(initial_state) != player.state_size:
            print(f"⚠️ State size changed from {player.state_size} to {len(initial_state)}. Updating player...")
            player.state_size = len(initial_state)
            player.model = None  # Force rebuild of the model on next move
        
        observations, result = simulate_maze_game(
            maze=env.maze,
            player=player,
            initial_position=env.start_pos,
            possible_actions=possible_actions,
            converted_directions=converted_directions,
            codes=codes,
            max_steps=max_steps,
            verbose=verbose
        )
        
        if episode % 100 == 0:
            player.update_target_model()
            if verbose:
                print(f"\n📊 Episode {episode}: Our apprentice {'mastered this maze' if result == 'win' else 'is still perfecting their skills'}.")
                print(f"   Current exploration rate: {player.epsilon:.2f}")
                print(f"   Memory size: {len(player.memory)} experiences")
                print(f"   Current state size: {player.state_size}")
    
    if verbose:
        print("\n🎉 Graduation Ceremony! 🎖️")
        print(f"Our Deep Q-Network apprentice has completed {num_episodes} rigorous training sessions.")
        print(f"Final memory size: {len(player.memory)} experiences")
        print(f"Final state size: {player.state_size}")
        print("Time to see how our AI performs in unseen mazes!")
    
    return player

def train_ppo_player(env, player, num_episodes=1000, max_steps=500, show_training=False, verbose=False):
    print("🚀 Welcome to the PPO Space Academy! 🌌")
    print(f"Our aspiring space navigator is preparing for {num_episodes} simulated space-maze missions.")
    
    # Ensure we have all necessary parameters from the environment
    possible_actions = env.get_possible_actions()
    converted_directions = env.get_converted_directions()
    codes = env.get_codes()

    # TODO MAYBE I DO NOT NEED THIS FOR THE PPO

    print("🛸 Initiating space-maze simulations!")
    for episode in tqdm(range(num_episodes), desc="Training Progress", disable=not show_training):
        env.reset()

        observations, result = simulate_maze_game(
            maze=env.maze,
            player=player,
            initial_position=env.start_pos,
            possible_actions=possible_actions,
            converted_directions=converted_directions,
            codes=codes,
            max_steps=max_steps,
            verbose=verbose
        )

        states, actions, rewards, next_states, dones = zip(*observations)
        player.update(states, actions, rewards, next_states, dones)
        
        if verbose and episode % 100 == 0:
            print(f"\n📊 Mission {episode}: Our navigator {'successfully charted the space-maze' if result == 'win' else 'is improving their navigation skills'}.")
            print(f"   Current mission success rate: {sum(1 for obs in observations if obs[2] > 0) / len(observations):.2f}")
    
    if verbose:
        print("\n🎉 Space Navigation Certification Achieved! 🏅")
        print(f"Our PPO space navigator has completed {num_episodes} challenging space-maze missions.")
        print("Time to explore the vast unknown mazes of the universe!")
    
    return player

# Example usage
if __name__ == "__main__":
    maze_size = (10, 10)
    num_episodes = 1000
    max_steps = 500

    print("Choose your learning algorithm:")
    print("1. Q-Learning")
    print("2. Deep Q-Network")
    print("3. Proximal Policy Optimization")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == "1":
        trained_player = train_q_player(maze_size, num_episodes, max_steps)
    elif choice == "2":
        trained_player = train_improved_q_player(maze_size, num_episodes, max_steps)
    elif choice == "3":
        trained_player = train_ppo_player(maze_size, num_episodes, max_steps)
    else:
        print("Invalid choice. Please run the script again and choose 1, 2, or 3.")
        exit()
    
    print("\nTraining complete! Your AI is ready for new adventures.")