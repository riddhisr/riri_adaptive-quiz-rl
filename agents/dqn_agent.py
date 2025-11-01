# agents/dqn_agent.py
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNetwork(nn.Module):
    def __init__(self, n_in, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, n_state=4, n_actions=3, lr=1e-3, gamma=0.99, buffer_size=10000):
        self.n_actions = n_actions
        self.gamma = gamma
        self.net = DQNetwork(n_state, n_actions).to(DEVICE)
        self.target = DQNetwork(n_state, n_actions).to(DEVICE)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.replay = deque(maxlen=buffer_size)
        self.batch_size = 64
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.update_target_steps = 500
        self.step_count = 0

    def select(self, state):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = self.net(state_t)
            return int(q.argmax().item())

    def store(self, s,a,r,ns,done):
        self.replay.append((s,a,r,ns,done))
        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target.load_state_dict(self.net.state_dict())

    def sample(self):
        import random
        batch = random.sample(self.replay, min(len(self.replay), self.batch_size))
        s,a,r,ns,done = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(done)

    def learn(self):
        if len(self.replay) < 128:
            return
        s,a,r,ns,done = self.sample()
        s_v = torch.FloatTensor(s).to(DEVICE)
        ns_v = torch.FloatTensor(ns).to(DEVICE)
        a_v = torch.LongTensor(a).to(DEVICE)
        r_v = torch.FloatTensor(r).to(DEVICE)
        done_mask = torch.BoolTensor(done).to(DEVICE)

        q_vals = self.net(s_v).gather(1, a_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            q_next = self.target(ns_v).max(1)[0]
            q_next[done_mask] = 0.0
        expected = r_v + self.gamma * q_next

        loss = nn.MSELoss()(q_vals, expected)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # decay eps
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def save(self, path):
        torch.save({'net': self.net.state_dict(), 'target': self.target.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location=DEVICE)
        self.net.load_state_dict(d['net'])
        self.target.load_state_dict(d['target'])
