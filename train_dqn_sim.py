# train_dqn_sim.py
import numpy as np
from agents.dqn_agent import DQNAgent
import random
import os
from tqdm import trange

# Simple simulated student environment
class SimStudentEnv:
    def __init__(self, max_q=5):
        self.max_q = max_q
        self.reset()
    def reset(self):
        self.q_no = 0
        # state: [prev_correct, time_norm, q_frac, last_action]
        self.prev_correct = 0.0
        self.time_norm = 0.2
        self.last_action = 1.0
        state = np.array([self.prev_correct, self.time_norm, 0.0, self.last_action], dtype=np.float32)
        return state
    def step(self, action):
        # action: 0=easy,1=medium,2=hard
        self.q_no += 1
        # difficulty increases chance of being wrong
        base_prob_correct = {0:0.9, 1:0.7, 2:0.5}[action]
        # student skill drifts slightly with q_no
        skill = max(0.2, 1.0 - (self.q_no * 0.02))
        prob_correct = base_prob_correct * skill
        correct = random.random() < prob_correct
        reward = (1.0 if correct else -1.0) * (0.5 if action==0 else 1.0 if action==1 else 1.5)
        time_taken = np.clip(np.random.normal(5.0 + action*2.0, 1.0), 0.2, 30.0)
        self.prev_correct = 1.0 if correct else 0.0
        self.time_norm = min(time_taken / 20.0, 1.0)
        self.last_action = float(action)
        q_frac = self.q_no / float(self.max_q)
        state = np.array([self.prev_correct, self.time_norm, q_frac, self.last_action], dtype=np.float32)
        done = (self.q_no >= self.max_q)
        info = {"correct": correct, "time_taken": time_taken}
        return state, reward, done, info

def train_sim(episodes=2000):
    env = SimStudentEnv(max_q=5)
    agent = DQNAgent(state_dim=4, action_dim=3)
    for ep in range(1, episodes+1):
        s = env.reset()
        total = 0.0
        for t in range(1, 100):
            a = agent.select(s)
            ns, r, done, info = env.step(a)
            agent.store(s, a, r, ns, done)
            agent.learn()
            s = ns
            total += r
            if done:
                break
        if ep % 50 == 0:
            print(f"Episode {ep} reward {total:.2f} eps {agent.epsilon():.3f}")
            agent.save()
    agent.save()
    print("Training complete. Model saved to models/dqn.pth")

if __name__ == "__main__":
    train_sim(episodes=2000)
