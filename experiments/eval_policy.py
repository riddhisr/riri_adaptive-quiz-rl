# # experiments/eval_policy.py
# from env.quiz_env import QuizEnv
# from agents.dqn_agent import DQNAgent
# import numpy as np

# agent = DQNAgent()
# agent.load('models/dqn.pth')
# agent.eps = 0.0  # greedy
# env = QuizEnv(max_q=10)
# N = 200
# rewards = []
# for _ in range(N):
#     s = env.reset()
#     done = False
#     tot = 0
#     while not done:
#         a = agent.select(s)
#         s, r, done, _ = env.step(a)
#         tot += r
#     rewards.append(tot)
# print('Eval reward mean/std:', np.mean(rewards), np.std(rewards))

# experiments/eval_policy_detailed.py
from env.quiz_env import QuizEnv
from agents.dqn_agent import DQNAgent
import numpy as np
from collections import Counter

def evaluate(model_path='models/dqn.pth', episodes=500, max_q=10, profile=None):
    agent = DQNAgent()
    agent.load(model_path)
    agent.eps = 0.0  # greedy for evaluation

    env = QuizEnv(user_profile=profile, max_q=max_q)
    rewards = []
    correct_counts = []
    difficulty_counts = Counter()

    for _ in range(episodes):
        s = env.reset()
        done = False
        tot = 0.0
        corrects = 0
        while not done:
            a = agent.select(s)
            ns, r, done, info = env.step(a)
            tot += r
            if info.get('correct'):
                corrects += 1
            difficulty_counts[a] += 1
            s = ns
        rewards.append(tot)
        correct_counts.append(corrects / float(max_q))

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_acc = np.mean(correct_counts)
    print(f"Model: {model_path} | Profile: {profile} | Episodes: {episodes}")
    print(f"Mean reward: {mean_reward:.3f}  Std: {std_reward:.3f}")
    print(f"Mean accuracy (per-episode): {mean_acc*100:.2f}%")
    total_actions = sum(difficulty_counts.values())
    print("Difficulty distribution (fraction):")
    for a in sorted(difficulty_counts.keys()):
        print(f"  action {a} ({'easy' if a==0 else 'medium' if a==1 else 'hard'}): "
              f"{difficulty_counts[a]} ({difficulty_counts[a]/total_actions:.2%})")

if __name__ == '__main__':
    # default eval using all profiles randomized
    evaluate(model_path='models/dqn.pth', episodes=500, max_q=10, profile=None)
    # evaluate per-profile:
    # evaluate(model_path='models/dqn.pth', episodes=500, max_q=10, profile='novice')
    # evaluate(model_path='models/dqn.pth', episodes=500, max_q=10, profile='expert')
