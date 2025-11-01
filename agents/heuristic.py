# agents/heuristic.py
import random

class HeuristicAgent:
    def __init__(self, eps=0.2):
        self.eps = eps
        self.last_result = None

    def select(self, state):
        # state: [last_result, time_norm, q_no_norm, difficulty]
        if self.last_result is None:
            return 1
        if self.last_result == 1:
            # go harder with prob 1-eps
            if random.random() > self.eps:
                return 2
            return 1
        else:
            return 0

    def observe(self, result):
        self.last_result = 1 if result else 0
