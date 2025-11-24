import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------
# Network
# -------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.model(x)

# -------------------
# Replay Buffer
# -------------------
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch):
        data = random.sample(self.buffer, batch)
        s, a, r, s2, d = zip(*data)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(s2),
            np.array(d)
        )

    def __len__(self):
        return len(self.buffer)

# -------------------
# DQN Agent
# -------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99,
                 lr=1e-3, batch_size=64, eps_start=1.0,
                 eps_end=0.05, eps_decay=10000,
                 target_update=1000):

        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.total_steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        self.target_update = target_update

    def choose_action(self, state):
        self.total_steps += 1

        # eps decay
        self.eps = max(
            self.eps_end,
            self.eps_start - self.total_steps / self.eps_decay
        )

        if random.random() < self.eps:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q(state)
        return q_vals.argmax().item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_values = self.q(s).gather(1, a)

        with torch.no_grad():
            next_q = self.q_target(s2).max(1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q_values, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.total_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return loss.item()
