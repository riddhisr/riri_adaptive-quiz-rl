# agents/tabular_q.py
import numpy as np
import random

class TabularQAgent:
    def __init__(self, lr=0.1, gamma=0.9, eps=1.0, eps_decay=0.995):
        # we'll discretize state to (last_result:0/1, difficulty:0/1/2)
        self.q = np.zeros((2,3,3))  # last_result x q_no_bucket x action
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay

    def _state_to_idx(self, state):
        last = int(state[0])
        q_no = min(2, int(state[2] * 3))
        return last, q_no

    def select(self, state):
        last, q_no = self._state_to_idx(state)
        if random.random() < self.eps:
            return random.choice([0,1,2])
        return int(self.q[last, q_no].argmax())

    def update(self, s, a, r, ns, done):
        last, q_no = self._state_to_idx(s)
        nlast, nq_no = self._state_to_idx(ns)
        best_next = 0 if done else self.q[nlast, nq_no].max()
        td = r + self.gamma * best_next - self.q[last,q_no,a]
        self.q[last,q_no,a] += self.lr * td
        if done:
            self.eps *= self.eps_decay
