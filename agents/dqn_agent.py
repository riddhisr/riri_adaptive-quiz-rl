# agents/dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from pathlib import Path

MODEL_PATH = Path("models/dqn.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buf)

class QNetwork(nn.Module):
    def __init__(self, in_dim=4, hidden=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim=4, action_dim=3, lr=1e-3, gamma=0.99, batch_size=64, buffer_size=50000, tau=1e-3):
        self.device = DEVICE
        self.q = QNetwork(in_dim=state_dim, out_dim=action_dim).to(self.device)
        self.q_target = QNetwork(in_dim=state_dim, out_dim=action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.action_dim = action_dim
        self.tau = tau
        self.total_steps = 0
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 20000

    def select(self, state, eval_mode=False):
        eps = self.epsilon()
        if eval_mode or random.random() > eps:
            try:
                with torch.no_grad():
                    x = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                    qvals = self.q(x)
                    a = int(qvals.argmax(dim=1).item())
                return a
            except Exception:
                return int(random.randrange(self.action_dim))
        else:
            return int(random.randrange(self.action_dim))

    def epsilon(self):
        return max(self.eps_end, self.eps_start - (self.total_steps / self.eps_decay) * (self.eps_start - self.eps_end))

    def store(self, s, a, r, ns, done):
        self.buffer.push(np.array(s, dtype=np.float32), int(a), float(r), np.array(ns, dtype=np.float32), bool(done))
        self.total_steps += 1

    def learn(self, updates=1):
        if len(self.buffer) < self.batch_size:
            return
        for _ in range(updates):
            batch = self.buffer.sample(self.batch_size)
            states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
            actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
            dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

            q_values = self.q(states).gather(1, actions)
            with torch.no_grad():
                q_next = self.q_target(next_states).max(1)[0].unsqueeze(1)
                q_target = rewards + (1.0 - dones) * (self.gamma * q_next)

            loss = nn.functional.mse_loss(q_values, q_target)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            for p, p_target in zip(self.q.parameters(), self.q_target.parameters()):
                p_target.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_target.data)

    def save(self, path=MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q.state_dict(), path)
    def load(self, path=MODEL_PATH):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target.load_state_dict(self.q.state_dict())
