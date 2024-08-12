import numpy as np
from typing import Union, Tuple, List, Dict, Any
from collections import defaultdict
import torch
from miniReinforcedMaze.agent.actions import max_visibility, apply_action
from miniReinforcedMaze.agent.players import QPlayer, ImprovedQPlayer, PPOPlayer
from miniReinforcedMaze.environment.generate_maze import RenderMaze

def evaluate_action_quality(old_position: Tuple[int, int], new_position: Tuple[int, int], reward: float, maze: np.ndarray, codes: Dict[str, int]) -> Tuple[str, str]:
    if maze[new_position] == codes["TREASURE"]:
        return "excellent", "💎 Excellent move! Found the treasure!"
    elif maze[new_position] == codes["HOLE"]:
        return "very bad", "💥 Oh no! Fell into a trap!"
    elif reward > 0:
        return "good", "👍 Good move! Getting closer to the goal."
    elif reward < 0:
        if new_position == old_position:
            return "bad", "🚫 Bad move! Hit a wall."
        else:
            return "bad", "👎 Bad move! Moving away from the goal or revisiting."
    else:
        return "neutral", "😐 Neutral move. Keep exploring!"

def simulate_maze_game(
    maze: np.ndarray,
    player: Union[QPlayer, ImprovedQPlayer, PPOPlayer],
    initial_position: Tuple[int, int] = (0, 0),
    possible_actions: Dict[str, int] = None,
    converted_directions: Dict[int, str] = None,
    reward_step: float = 0.0,
    reward_repeat: float = -0.1,
    reward_lose: float = -1.0,
    reward_win: float = 1.0,
    codes: Dict[str, int] = None,
    cropped_images: List[Any] = None,
    verbose: bool = False,
    show_gif: bool = False,
    max_steps: int = 1000
) -> Tuple[List[Tuple[Any, int, float, Any, bool]], str]:
    
    if maze is None or player is None or possible_actions is None or converted_directions is None or codes is None:
        raise ValueError("maze, player, possible_actions, converted_directions, and codes must be provided")

    print("🌟 A new maze adventure begins!")
    print(f"🚶 Our hero starts at position {initial_position}")

    position = initial_position
    observations = []
    memory = defaultdict(int)
    memory[initial_position] = 1

    if show_gif:
        gif_maze = RenderMaze()
        gif_maze.add(maze=maze.copy(), position=position, cropped_images=cropped_images)
        print("🎥 Recording the adventure for posterity...")

    for turn_counter in range(1, max_steps + 1):
        print(f"\n🕰️ Turn {turn_counter}")
        
        # Create the state representation
        state = np.array(
            position
            + max_visibility(maze=maze, position=position, max_distance=5)
            + tuple(
                memory[apply_action(position=position, maze=maze, possible_actions=possible_actions, action=action)]
                for _, action in possible_actions.items()
            )
        ).reshape(1, -1)
        
        print(f"State shape: {state.shape}")
        print(f"State content: {state}")

        print("🤔 Our hero ponders the next move...")
        action = player.move(state)
        print(f"💡 Decision made: {converted_directions[action]}")

        try:
            new_position = apply_action(position=position, maze=maze, possible_actions=possible_actions, action=action)
            print(f"🚶 Our hero moves to {new_position}")
        except Exception as e:
            print(f"❌ Oops! Something went wrong: {str(e)}")
            print(f"🤖 Player details: {player}")
            raise

        # Calculate reward
        if memory[new_position] == 1:
            reward = reward_repeat
            print("🔄 Uh-oh! Our hero has been here before. That's not ideal.")
        else:
            reward = reward_step
            print("🆕 Exploring new territory!")

        memory[new_position] = 1

        # Create the new state representation
        new_state = np.array(
            new_position
            + max_visibility(maze=maze, position=new_position, max_distance=5)
            + tuple(
                memory[apply_action(position=new_position, maze=maze, possible_actions=possible_actions, action=action)]
                for _, action in possible_actions.items()
            )
        ).reshape(1, -1)

        print(f"💰 Reward for this move: {reward}")

        # Check for game-ending conditions
        done = False
        if maze[new_position] == codes["HOLE"]:
            reward = reward_lose
            result = "lose"
            done = True
            print("💀 Oh no! Our hero fell into a hole. Game over!")
        elif maze[new_position] == codes["TREASURE"]:
            reward = reward_win
            result = "win"
            done = True
            print("🏆 Hooray! Our hero found the treasure. Victory!")

        observations.append((state.squeeze(), action, reward, new_state.squeeze(), done))

        # Update the player
        if isinstance(player, QPlayer):
            player.update(state.squeeze(), action, reward, new_state.squeeze())
        elif isinstance(player, ImprovedQPlayer):
            player.update(state.squeeze(), action, reward, new_state.squeeze(), done)
        elif isinstance(player, PPOPlayer):
            player.update([state.squeeze()], [action], [reward], [new_state.squeeze()], [done])

        if done:
            break

        position = new_position

        if verbose:
            print_verbose_output(
                turn_counter=turn_counter,
                position=position,
                old_state=state.squeeze(),
                action=action,
                converted_directions=converted_directions,
                player=player,
                possible_actions=possible_actions,
                new_position=new_position,
                new_state=new_state.squeeze(),
                reward=reward,
                maze=maze
            )

        if show_gif:
            gif_maze.add(maze=maze.copy(), position=position, cropped_images=cropped_images)

    if 'result' not in locals():
        result = "timeout"
        print("⏰ Time's up! Our hero couldn't find the treasure in time.")

    if verbose:
        print_verbose_output(
            turn_counter=turn_counter,
            position=position,
            old_state=state.squeeze(),
            action=action,
            converted_directions=converted_directions,
            player=player,
            possible_actions=possible_actions,
            new_position=new_position,
            new_state=new_state.squeeze(),
            reward=reward,
            maze=maze,
            result=result,
            observations=observations
        )

    if show_gif:
        gif_maze.show(possible_actions=possible_actions)
        print("🎬 The adventure has been recorded! You can now watch the replay.")

    print(f"\n🏁 Adventure summary:")
    print(f"   Outcome: {result}")
    print(f"   Steps taken: {turn_counter}")
    print(f"   Final position: {position}")
    print(f"   Total reward: {sum(obs[2] for obs in observations)}")

    return observations, result

def print_verbose_output(**kwargs):
    print("\n📊 Verbose Output:")
    print(f"   Turn: {kwargs['turn_counter']}")
    print(f"   Position: {kwargs['position']}")
    print(f"   Action: {kwargs['converted_directions'][kwargs['action']]}")
    print(f"   New Position: {kwargs['new_position']}")
    print(f"   Reward: {kwargs['reward']}")
    
    if 'result' in kwargs:
        print(f"   Final Result: {kwargs['result']}")
        print(f"   Total Steps: {len(kwargs['observations'])}")
        print(f"   Total Reward: {sum(obs[2] for obs in kwargs['observations'])}")
    
    if isinstance(kwargs['player'], QPlayer):
        print(f"   Q-value for this state-action: {kwargs['player'].Q.get((tuple(kwargs['old_state']), kwargs['action']), 0)}")
    elif isinstance(kwargs['player'], ImprovedQPlayer):
        print("   DQN Update: Updating neural network...")
    elif isinstance(kwargs['player'], PPOPlayer):
        print("   PPO Update: Collecting experience for policy update...")
    
    print(f"   State: {kwargs['old_state']}")
    print(f"   New State: {kwargs['new_state']}")
    print("=" * 50)