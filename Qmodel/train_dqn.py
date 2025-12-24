"""
train_dqn.py - Deep Q-Network Training for Antenna Optimization
================================================================

This implements a PROPER Deep Q-Learning algorithm following the original
DQN paper (Mnih et al., 2015) with improvements.

KEY RL CONCEPTS IMPLEMENTED:
----------------------------
1. Markov Decision Process (MDP):
   - State: Current antenna config + target angle + current metrics
   - Action: Modify configuration (add/remove elements from rings)
   - Reward: RF performance (gain, SSL, HPBW)
   - Transition: Physics simulation of new configuration

2. Deep Q-Network:
   - Neural network Q(s,a) approximates action-value function
   - Predicts expected cumulative reward for each action
   - Updated using Bellman equation: Q(s,a) = r + gamma * max_a' Q(s',a')

3. Experience Replay:
   - Stores transitions (s, a, r, s', done) in replay buffer
   - Samples random mini-batches to break temporal correlations
   - Stabilizes training

4. Target Network:
   - Separate network for computing target Q-values
   - Updated periodically (not every step)
   - Prevents oscillations in Q-value estimates

5. Epsilon-Greedy Exploration:
   - Starts with high exploration (epsilon=1.0)
   - Gradually shifts to exploitation (epsilon->0.05)
   - Balances exploration vs exploitation

Output files:
- dqn_model.pth: Trained model weights
- reward_history.npy/.mat: Training rewards
- training_curve.png: Learning curve visualization
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import time
import matplotlib.pyplot as plt
from scipy.io import savemat

from env_antenna import AntennaEnvOnline


def clear_console():
    print("\n" + "=" * 80)
    print(" " * 25 + "DQN ANTENNA OPTIMIZATION")
    print("=" * 80 + "\n")


# =============================================================================
# Q-Network Architecture
# =============================================================================
class QNetwork(nn.Module):
    """
    Deep Q-Network for action-value function approximation.
    
    Architecture:
    - Input: State vector (9D: config + theta0 + metrics)
    - Hidden: 2 fully connected layers with ReLU activation
    - Output: Q-value for each action (11 actions)
    
    The network learns: Q(s, a) = Expected cumulative reward starting from
    state s, taking action a, and following optimal policy thereafter.
    """
    
    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim)
        )
        
        # Initialize weights properly
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# Experience Replay Buffer
# =============================================================================
class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    
    Stores transitions (s, a, r, s', done) and samples random mini-batches.
    This breaks temporal correlations and stabilizes training.
    
    Why it's important:
    - Sequential experiences are correlated (bad for SGD)
    - Random sampling provides i.i.d. training data
    - Allows reuse of rare/important experiences
    """
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random mini-batch."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# DQN Agent
# =============================================================================
class DQNAgent:
    """
    Deep Q-Network Agent.
    
    Implements the full DQN algorithm:
    1. Epsilon-greedy action selection
    2. Experience replay
    3. Target network for stable Q-targets
    4. Bellman equation updates
    
    The Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    
    In DQN, this becomes minimizing the loss:
        L = (r + gamma * max_a' Q_target(s',a') - Q(s,a))^2
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,           # Discount factor
        lr=5e-4,              # Learning rate
        batch_size=64,        # Mini-batch size
        eps_start=1.0,        # Initial exploration
        eps_end=0.05,         # Final exploration
        eps_decay_steps=50000,  # Steps to decay epsilon
        target_update_freq=1000,  # Steps between target network updates
        buffer_size=100000,   # Replay buffer size
        hidden1=256,          # First hidden layer size
        hidden2=128           # Second hidden layer size
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Epsilon-greedy parameters
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        
        self.target_update_freq = target_update_freq
        self.total_steps = 0
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Q-Network (online network - updated every step)
        self.q_network = QNetwork(state_dim, action_dim, hidden1, hidden2).to(self.device)
        
        # Target Network (updated periodically for stable targets)
        self.target_network = QNetwork(state_dim, action_dim, hidden1, hidden2).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is never trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.loss_history = []
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon: random action (exploration)
        With probability 1-epsilon: best action according to Q (exploitation)
        """
        self.total_steps += 1
        
        # Decay epsilon linearly
        if training:
            self.eps = max(
                self.eps_end,
                self.eps_start - (self.eps_start - self.eps_end) * 
                (self.total_steps / self.eps_decay_steps)
            )
            
            # Exploration: random action
            if random.random() < self.eps:
                return random.randrange(self.action_dim)
        
        # Exploitation: best action according to Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def update(self):
        """
        Perform one step of Q-learning update.
        
        1. Sample mini-batch from replay buffer
        2. Compute current Q-values: Q(s, a)
        3. Compute target Q-values: r + gamma * max_a' Q_target(s', a')
        4. Compute loss and backpropagate
        5. Periodically update target network
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q-values: Q(s, a) for the actions that were taken
        current_q = self.q_network(states).gather(1, actions)
        
        # Target Q-values using target network (Double DQN style)
        with torch.no_grad():
            # Use online network to select actions
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # Use target network to evaluate those actions
            next_q = self.target_network(next_states).gather(1, next_actions)
            # Bellman target: r + gamma * Q(s', argmax_a' Q(s', a'))
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss (Huber loss is more robust than MSE)
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network periodically
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.loss_history.append(loss.item())
        return loss.item()
    
    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'eps': self.eps,
            'loss_history': self.loss_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.eps = checkpoint['eps']
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        print(f"Model loaded from {filepath}")


# =============================================================================
# Training Function
# =============================================================================
def train_dqn(episodes=3000, max_steps=50, seed=42):
    """
    Train the DQN agent for antenna optimization.
    
    Training loop:
    1. Reset environment with random target angle
    2. Agent takes actions, receives rewards
    3. Store experiences in replay buffer
    4. Update Q-network using mini-batch gradient descent
    5. Repeat until convergence
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("=" * 60)
    print("DEEP Q-NETWORK TRAINING")
    print("=" * 60)
    print("\nThis is a PROPER RL implementation:")
    print("- State includes current config + target + metrics")
    print("- Actions incrementally modify the antenna configuration")
    print("- Agent learns Q(s,a) = expected cumulative reward")
    print("- No dataset needed - learns through simulation")
    print()
    
    # Create environment
    env = AntennaEnvOnline(
        max_rings=5,
        max_elements=10,
        max_steps=max_steps,
        theta0_range=(30, 150),
        seed=seed
    )
    
    state_dim = env.state_dim  # 9
    action_dim = env.action_dim  # 11
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print()
    
    # Create DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=5e-4,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=episodes * max_steps // 2,
        target_update_freq=500,
        hidden1=256,
        hidden2=128
    )
    
    # Training statistics
    reward_history = []
    best_avg_reward = -np.inf
    best_episode = 0
    
    start_time = time.time()
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.buffer.push(state, action, reward, next_state, done)
            
            # Update Q-network
            agent.update()
            
            # Accumulate reward
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        reward_history.append(episode_reward)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            elapsed = time.time() - start_time
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_episode = episode + 1
            
            print(f"Episode {episode + 1:4d}/{episodes} | "
                  f"Reward: {episode_reward:.3f} | "
                  f"Avg(100): {avg_reward:.3f} | "
                  f"Epsilon: {agent.eps:.3f} | "
                  f"Buffer: {len(agent.buffer):6d} | "
                  f"Time: {elapsed:.1f}s")
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {training_time:.1f} seconds")
    print(f"Best average reward: {best_avg_reward:.4f} (Episode {best_episode})")
    print(f"Final epsilon: {agent.eps:.4f}")
    print(f"Total steps: {agent.total_steps}")
    
    return agent, env, reward_history


# =============================================================================
# Main Execution
# =============================================================================
def main():
    clear_console()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Training configuration
    EPISODES = 3000
    MAX_STEPS = 50
    SEED = 42
    
    print("Configuration:")
    print(f"  Episodes: {EPISODES}")
    print(f"  Max steps/episode: {MAX_STEPS}")
    print(f"  Seed: {SEED}")
    print()
    
    # Train the agent
    agent, env, reward_history = train_dqn(
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        seed=SEED
    )
    
    # Save model and results
    model_path = os.path.join(script_dir, 'dqn_model.pth')
    agent.save(model_path)
    
    # Save reward history
    reward_history = np.array(reward_history, dtype=np.float32)
    np.save(os.path.join(script_dir, 'reward_history.npy'), reward_history)
    savemat(os.path.join(script_dir, 'reward_history.mat'), {
        'reward_history': reward_history
    })
    
    # Save environment parameters for testing
    env_params = {
        'state_dim': env.state_dim,
        'action_dim': env.action_dim,
        'max_rings': env.max_rings,
        'max_elements': env.max_elements,
        'theta0_range': env.theta0_range,
        'ml_range': env.ml_range,
        'ssl_range': env.ssl_range,
        'hpbw_range': env.hpbw_range
    }
    np.savez(os.path.join(script_dir, 'env_params.npz'), **env_params)
    
    print("\nFiles saved:")
    print("  - dqn_model.pth (trained model)")
    print("  - reward_history.npy (training rewards)")
    print("  - reward_history.mat (MATLAB format)")
    print("  - env_params.npz (environment parameters)")
    
    # Plot training curve
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Raw rewards with moving average
    plt.subplot(1, 2, 1)
    plt.plot(reward_history, alpha=0.3, color='blue', label='Episode reward')
    
    window = 100
    if len(reward_history) >= window:
        moving_avg = np.convolve(reward_history, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(reward_history)), moving_avg,
                 color='red', linewidth=2, label=f'Moving avg ({window} eps)')
    
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('DQN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Loss curve
    plt.subplot(1, 2, 2)
    if agent.loss_history:
        loss_smooth = np.convolve(agent.loss_history, 
                                   np.ones(100) / 100, mode='valid')
        plt.plot(loss_smooth, color='green', linewidth=1)
        plt.xlabel('Update Step')
        plt.ylabel('Loss (Smoothed)')
        plt.title('Q-Network Loss')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'training_curve.png'), dpi=300)
    plt.show()
    
    print("\n  - training_curve.png (learning curves)")
    print("\nTraining complete! Run test_dqn.py to evaluate the model.")


if __name__ == "__main__":
    main()
