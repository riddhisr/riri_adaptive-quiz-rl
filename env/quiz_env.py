# env/quiz_env.py
import numpy as np
from utils.sim_user import SimUser

# A small Gym-like environment wrapper for quiz interactions
class QuizEnv:
    def __init__(self, user_profile=None, max_q=10):
        # action space: 0=easy,1=medium,2=hard
        self.action_space = [0,1,2]
        # observation: [last_result(0/1), time_taken_norm, q_no_norm, difficulty]
        self.max_q = max_q
        self.q_no = 0
        self.user = SimUser(profile=user_profile)
        self.state = None

    def reset(self):
        self.q_no = 0
        self.user.reset()
        # initial state: last_result=0, time=0.5, q_no=0, difficulty=1 (medium)
        self.state = np.array([0.0, 0.5, 0.0, 1.0], dtype=np.float32)
        return self.state

    def step(self, action):
        # action in {0,1,2}
        difficulty = action
        # SimUser returns (correct_bool, time_taken_seconds)
        correct, time_taken = self.user.answer_question(difficulty)
        self.q_no += 1
        done = self.q_no >= self.max_q

        # reward design
        base = 1.0 if correct else -1.0
        diff_factor = {0:0.5, 1:1.0, 2:1.5}[difficulty]
        time_pen = min(time_taken / 10.0, 1.0)
        reward = base * diff_factor - 0.5 * time_pen

        # next state
        last_result = 1.0 if correct else 0.0
        time_norm = min(time_taken / 10.0, 1.0)
        q_no_norm = self.q_no / float(self.max_q)
        self.state = np.array([last_result, time_norm, q_no_norm, float(difficulty+1)], dtype=np.float32)

        info = {"correct": correct, "time_taken": time_taken}
        return self.state, reward, done, info
