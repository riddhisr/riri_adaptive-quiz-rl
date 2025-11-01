# experiments/retrain_from_logs.py
"""
Simple retraining script that consumes data/transitions.csv (if present) and fine-tunes a DQN.
If transitions.csv not present, it falls back to training from simulated env.
"""
import argparse, os, csv, numpy as np
from env.quiz_env import QuizEnv
from agents.dqn_agent import DQNAgent

def load_transitions(path):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = np.array([float(row["s0"]), float(row["s1"]), float(row["s2"]), float(row["s3"])], dtype=np.float32)
            a = int(row["a"])
            r = float(row["r"])
            ns = np.array([float(row["ns0"]), float(row["ns1"]), float(row["ns2"]), float(row["ns3"])], dtype=np.float32)
            done = bool(int(row["done"]))
            data.append((s,a,r,ns,done))
    return data

def train_from_transitions(agent, transitions, epochs=5):
    # naive training: store into replay buffer and run learn steps
    for s,a,r,ns,done in transitions:
        agent.store(s,a,r,ns,done)
    for _ in range(epochs):
        agent.learn()
    return agent

def train_from_scratch(episodes=2000, max_q=10, save_path="models/dqn_from_scratch.pth"):
    env = QuizEnv(user_profile=None, max_q=max_q)
    agent = DQNAgent()
    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = agent.select(s)
            ns, r, done, info = env.step(a)
            agent.store(s,a,r,ns,done)
            agent.learn()
            s = ns
    agent.save(save_path)
    print("Saved:", save_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/transitions.csv")
    p.add_argument("--save", type=str, default="models/dqn_retrained.pth")
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    if os.path.exists(args.input):
        print("Loading transitions from", args.input)
        trans = load_transitions(args.input)
        agent = DQNAgent()
        agent = train_from_transitions(agent, trans, epochs=args.epochs)
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        agent.save(args.save)
        print("Saved retrained model to", args.save)
    else:
        print("No transitions.csv found. Training from scratch instead.")
        train_from_scratch(episodes=1000, save_path=args.save)
