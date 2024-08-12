import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class BasePlayer:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def move(self, state):
        raise NotImplementedError("Subclasses must implement the move method")

class QPlayer(BasePlayer):
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        super().__init__(state_size, action_size)
        self.Q = defaultdict(float)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay

    def move(self, state):
        state = tuple(state)  # Convert numpy array to tuple for dictionary key
        if random.random() > self.epsilon:
            q_values = [self.Q.get((state, a), 0.0) for a in range(self.action_size)]
            return np.argmax(q_values)
        else:
            return random.randint(0, self.action_size - 1)

    def update(self, state, action, reward, next_state):
        state, next_state = tuple(state), tuple(next_state)
        current_q = self.Q.get((state, action), 0.0)
        next_q = max(self.Q.get((next_state, a), 0.0) for a in range(self.action_size))
        self.Q[(state, action)] = current_q + self.lr * (reward + self.gamma * next_q - current_q)
        self.epsilon *= self.epsilon_decay

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class ImprovedQPlayer:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.memory = []
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.learning_rate = learning_rate
        print(f"ImprovedQPlayer initialized with state_size: {state_size}, action_size: {action_size}")

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def move(self, state):
        state = torch.FloatTensor(state).view(1, -1)
        if self.model is None or state.shape[1] != self.state_size:
            self.state_size = state.shape[1]
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            print(f"Model built/rebuilt with input size: {self.state_size}")
            print(f"Model architecture: {self.model}")
        
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.max(1)[1].item()
        else:
            return random.randint(0, self.action_size - 1)

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)
        
        if len(self.memory) > 32:
            batch = random.sample(self.memory, 32)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            
            current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
            loss = nn.MSELoss()(current_q, target_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.epsilon *= self.epsilon_decay
        if self.epsilon < 0.01:
            self.epsilon = 0.01

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

class PPOPlayer:
    def __init__(self, state_size, action_size, learning_rate=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.actor_critic = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def move(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute advantages
        _, next_values = self.actor_critic(next_states)
        _, values = self.actor_critic(states)
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = self.compute_gae(deltas.detach(), dones)

        # PPO update
        for _ in range(10):  # Number of optimization epochs
            action_probs, values = self.actor_critic(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            old_log_probs = dist.log_prob(actions).detach()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(values, rewards + self.gamma * next_values * (1 - dones))
            entropy = dist.entropy().mean()

            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_gae(self, rewards, dones, gamma=0.99, lam=0.95):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * gae * (1 - dones[step]) - gae
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + rewards[step])
        return torch.tensor(returns)
    
