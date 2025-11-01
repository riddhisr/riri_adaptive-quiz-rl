# # experiments/train_dqn.py
# import argparse
# import numpy as np
# import os
# from env.quiz_env import QuizEnv
# from agents.dqn_agent import DQNAgent
# import matplotlib.pyplot as plt
# import csv

# def train(episodes=2000, max_q=10, save_path=None, report_dir="experiments"):
#     os.makedirs(os.path.dirname(save_path) if save_path else "models", exist_ok=True)
#     os.makedirs(report_dir, exist_ok=True)

#     env = QuizEnv(user_profile=None, max_q=max_q)
#     agent = DQNAgent(n_state=4, n_actions=3)

#     rewards = []
#     for ep in range(episodes):
#         s = env.reset()
#         done = False
#         total = 0.0
#         while not done:
#             a = agent.select(s)
#             ns, r, done, info = env.step(a)
#             agent.store(s, a, r, ns, done)
#             agent.learn()
#             s = ns
#             total += r
#         rewards.append(total)
#         if (ep + 1) % 100 == 0 or ep == 0:
#             print(f"Episode {ep+1}/{episodes} - reward: {np.mean(rewards[-100:]):.3f} - eps: {agent.eps:.3f}")

#     # save rewards csv
#     csv_path = os.path.join(report_dir, "rewards.csv")
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["episode", "reward"])
#         for i, r in enumerate(rewards, 1):
#             writer.writerow([i, r])
#     print(f"Saved rewards CSV -> {csv_path}")

#     # plot and save
#     plt.figure(figsize=(8,4))
#     plt.plot(rewards, label="episode reward")
#     plt.xlabel('episode')
#     plt.ylabel('reward')
#     plt.title('Training rewards')
#     plt.grid(alpha=0.3)
#     plt.legend()
#     png_path = os.path.join(report_dir, "rewards.png")
#     plt.tight_layout()
#     plt.savefig(png_path)
#     print(f"Saved rewards plot -> {png_path}")

#     # save model
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         agent.save(save_path)
#         print(f"Saved model -> {save_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--episodes', type=int, default=2000)
#     parser.add_argument('--max_q', type=int, default=10)
#     parser.add_argument('--save', type=str, default='models/dqn.pth')
#     parser.add_argument('--report_dir', type=str, default='experiments')
#     args = parser.parse_args()
#     train(args.episodes, args.max_q, args.save, args.report_dir)

# experiments/train_dqn.py
import argparse, os, csv
import numpy as np
from env.quiz_env import QuizEnv
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

def train(episodes=2000, max_q=10, save_path="models/dqn.pth", report_dir="experiments"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    env = QuizEnv(user_profile=None, max_q=max_q)
    agent = DQNAgent()
    rewards = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.select(s)
            ns, r, done, info = env.step(a)
            agent.store(s,a,r,ns,done)
            agent.learn()
            s = ns
            total += r
        rewards.append(total)
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} - mean last100: {np.mean(rewards[-100:]):.3f} - eps: {agent.eps:.3f}")
    # save CSV
    csv_path = os.path.join(report_dir, "rewards.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode","reward"])
        for i,r in enumerate(rewards,1):
            w.writerow([i,r])
    # plot
    png_path = os.path.join(report_dir, "rewards.png")
    plt.figure(figsize=(8,4))
    plt.plot(rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.tight_layout()
    plt.savefig(png_path)
    # save model
    agent.save(save_path)
    print("Saved model to", save_path)
    print("Saved rewards CSV and PNG to", report_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max_q", type=int, default=10)
    p.add_argument("--save", type=str, default="models/dqn.pth")
    p.add_argument("--report_dir", type=str, default="experiments")
    args = p.parse_args()
    train(args.episodes, args.max_q, args.save, args.report_dir)
