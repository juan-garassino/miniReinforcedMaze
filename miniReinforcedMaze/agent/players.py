# import numpy as np
from miniReinforcedMaze.agent.actions import choose_action
from miniReinforcedMaze.generate.generate_maze import MazeEnvironment
from miniReinforcedMaze.agent.simulate import simulate_maze_game
import numpy as np
import random
from collections import defaultdict
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class Player(object):
    """
    Abstract base class for all player types.

    This class defines the interface for player objects in the game.
    All specific player types should inherit from this class.
    """

    def __init__(self):
        """
        Initialize the Player object.

        Attributes:
            i (int): A counter, possibly used for tracking moves or iterations.
        """
        self.i = 0

    def move(self, state):
        """
        Abstract method to determine the player's move.

        Args:
            state: The current state of the game.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclass responsibility")

class RandomPlayer(Player):
    """
    A player that makes random moves.

    This player simply chooses a random action from the available actions.
    """

    def move(self, state, possible_actions):
        """
        Choose a random move from the possible actions.

        Args:
            state: The current state of the game (unused in this implementation).
            posible_actions (list): A list of possible actions to choose from.

        Returns:
            The randomly chosen action.
        """
        return np.random.choice(possible_actions)

class QPlayer:
    def __init__(self, Q, explore_factor):
        self.Q = defaultdict(float)
        self.Q.update(Q)
        self.explore_factor = explore_factor
        print("🎭 A new Q-learning hero is born!")
        print(f"🧭 Initial exploration factor: {explore_factor}")
        print("📚 The hero's book of knowledge (Q-table) is mostly empty, ready to be filled with wisdom.")

    def move(self, state, possible_actions):
        print("\n🤔 Our hero faces a decision...")
        print(f"Current state: {state}")
        print(f"Possible actions: {possible_actions}")

        if random.random() > self.explore_factor:
            taken_action, current_value = choose_action(self.Q, possible_actions, state)
            print("🧠 The hero decides to use its wisdom!")
            print(f"Chosen action: {taken_action} (Q-value: {current_value})")
        else:
            taken_action = random.choice(possible_actions)
            print("🎲 The hero decides to explore!")
            print(f"Randomly chosen action: {taken_action}")

        return taken_action

def q_learning(Q, observations, learning_rate=0.2, discount_factor=0.95):
    print("\n📖 Time to update the hero's book of knowledge!")
    for old_state, action, reward, new_state in observations:
        print(f"\nReflecting on the experience: {old_state} -> {action} -> {new_state}")
        print(f"Reward received: {reward}")

        all_actions = set(action for (action, _) in Q.keys())
        _, estimated_future_value = choose_action(Q, list(all_actions), new_state)

        current_value = Q.get((action, old_state), 0)
        print(f"Current Q-value: {current_value}")
        print(f"Estimated future value: {estimated_future_value}")

        new_value = current_value + learning_rate * (
            reward + discount_factor * estimated_future_value - current_value)
        Q[(action, old_state)] = new_value

        print(f"Updated Q-value: {new_value}")

    return Q

def train_q_player(q_player, maze_size, maze_density, num_mazes, episodes_per_maze, **kwargs):
    print("🏰 The grand training adventure begins!")
    total_episodes = 0

    for maze_number in range(num_mazes):
        print(f"\n🗺️ Chapter {maze_number + 1}: A new maze appears!")
        maze_env = MazeEnvironment(maze_size[0], maze_size[1], density=maze_density)

        for episode in range(episodes_per_maze):
            total_episodes += 1
            print(f"\n🕯️ Episode {episode + 1} in maze {maze_number + 1} begins...")

            observations, result = simulate_maze_game(maze=maze_env.maze, player=q_player, **kwargs)

            old_Q = q_player.Q.copy()
            new_Q = q_learning(q_player.Q, observations)

            if total_episodes % 1000 == 0:
                convergence = sum([abs(old_Q[key] - new_Q[key]) for key in old_Q.keys()])
                print(f"📊 Progress check: Episode {total_episodes}, Convergence: {convergence:.2f}")

            q_player.Q = new_Q

        q_player.explore_factor = max(0.05, q_player.explore_factor * 0.99)
        print(f"🧭 The hero becomes a bit less exploratory. New explore factor: {q_player.explore_factor:.2f}")

    print(f"\n🎉 Training complete! Our hero has experienced {total_episodes} adventures across {num_mazes} mazes.")
    print(f"🧠 Final exploration factor: {q_player.explore_factor:.2f}")
    print("📚 The hero's book of knowledge is now filled with valuable insights!")

    return q_player

class ImprovedQPlayer:
    def __init__(self, action_size, learning_rate=0.001, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995):
        # Our AI hero is born, ready to tackle maze challenges!
        print("🐣 A new AI explorer is born!")
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.memory = deque(maxlen=2000)
        self.model = None
        print(f"🧭 Our hero can perform {action_size} different actions.")
        print(f"🧠 Learning rate: {learning_rate}, Discount factor: {discount_factor}")
        print(f"🎲 Initial exploration rate: {exploration_rate}")

    def _build_model(self, state_size):
        # Our hero's brain is forming - neural pathways are being created!
        print("🧠 Building neural pathways in our hero's brain...")
        model = Sequential([
            Input(shape=(state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        print("🎉 Neural network constructed! Our hero is ready to learn.")
        return model

    def remember(self, state, action, reward, next_state, done):
        # Our hero is storing a memory of its adventure
        self.memory.append((state, action, reward, next_state, done))
        print("💾 A new memory is stored in our hero's mind.")

    def act(self, state):
        print("🤔 Our hero is deciding: explore randomly or use learned knowledge?")
        if self.model is None:
            self.model = self._build_model(len(state))
        
        if np.random.rand() <= self.exploration_rate:
            action = random.randrange(self.action_size)
            print(f"🎲 Exploration wins! Choosing a random action: {action}")
            return action
        else:
            state = np.reshape(state, [1, -1])
            q_values = self.model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            print(f"🧠 Exploitation wins! Choosing the best known action: {action}")
            return action

    def move(self, state):
        # Alias for act method
        return self.act(state)

    def replay(self, batch_size):
        # Our hero is reflecting on past experiences
        print(f"📚 Time for reflection! Reviewing {batch_size} past adventures...")
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, -1])
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state, verbose=0)[0])
            state = np.reshape(state, [1, -1])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > 0.01:
            self.exploration_rate *= self.exploration_decay
        print(f"🧠 Lessons learned! New exploration rate: {self.exploration_rate:.4f}")

    def learn(self, observations):
        # Our hero is processing a batch of recent experiences
        print("📖 Time to learn from recent adventures!")
        for old_state, action, reward, new_state in observations:
            done = reward == 1.0 or reward == -1.0
            
            if self.model is None:
                self.model = self._build_model(len(old_state))
            
            self.remember(old_state, action, reward, new_state, done)
        
        if len(self.memory) > 32:
            print("🧠 Enough experiences gathered. Time for a deeper reflection...")
            self.replay(32)

        print(f"📊 Memory bank status: {len(self.memory)} adventures stored.")

def train_improved_q_player(
    player=None,
    maze_size=(10, 10),
    maze_density=0.2,
    episodes=1000,
    max_steps=500,
    batch_size=32,
    show_every=100,
    possible_actions=None,
    converted_directions=None,
    codes=None
):
    if possible_actions is None or converted_directions is None or codes is None:
        raise ValueError("possible_actions, converted_directions, and codes must be provided")

    print("\n🤖 The Tale of the Maze-Exploring AI 🤖")
    print("Our brave AI agent is about to embark on a grand adventure...")
    print(f"It will explore {episodes} different mazes, each up to {max_steps} steps long.")
    print("Let's see how it fares!\n")

    for e in range(episodes):
        env = MazeEnvironment(maze_size[0], maze_size[1], density=maze_density)
        
        print(f"\n📜 Chapter {e+1}: A New Maze Appears 📜")
        print("Our AI hero finds itself at the entrance of a mysterious new maze.")
        
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        print("The journey begins...\n")
        
        while not done and step < max_steps:
            action = player.act(state)
            next_state, reward, done = env.step(action)
            
            player.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step += 1
            
            if len(player.memory) > batch_size:
                print("🧠 Our hero takes a moment to reflect on its past experiences...")
                player.replay(batch_size)
                print("Lessons learned! The AI feels a bit wiser now.")
            
            if step % 10 == 0:
                print(f"Step {step}: Our hero continues its quest, driven by an exploration rate of {player.exploration_rate:.2f}")
        
        if done and reward > 0:
            print("\n🎉 Success! Our AI hero has found the treasure!")
        elif done:
            print("\n💀 Oh no! Our AI hero has fallen into a trap. Better luck next time!")
        else:
            print("\n⏱️ Time's up! Our hero couldn't find the treasure this time.")
        
        if e % show_every == 0:
            print(f"\n📊 Adventure Summary:")
            print(f"Episode: {e+1}/{episodes}")
            print(f"Steps taken: {step}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Exploration rate: {player.exploration_rate:.2f}")
            print("\nOur hero grows stronger with each maze...")
    
    print("\n🏆 The Training Saga is Complete! 🏆")
    print(f"After exploring {episodes} mazes, our AI hero has become a master explorer!")
    print(f"Final exploration rate: {player.exploration_rate:.4f}")
    print(f"Total experiences gathered: {len(player.memory)}")
    print("\nNow, let's see how well our hero performs in a new maze!")

    return player

def test_agent(env, agent, num_episodes=10, max_steps=500):
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            action = agent.act(state)
            next_state, done = env.step(action)
            reward = env.get_reward()
            next_state = np.reshape(next_state, [1, agent.state_size])
            state = next_state
            total_reward += reward
            step += 1
        
        print(f"Test Episode: {episode+1}/{num_episodes}, Steps: {step}, Total Reward: {total_reward:.2f}")