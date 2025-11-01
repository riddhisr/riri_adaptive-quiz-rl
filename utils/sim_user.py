# utils/sim_user.py
import random
import math

class SimUser:
    """
    Simulated user model to generate responses for training.
    Profiles: 'novice', 'intermediate', 'expert', or None (randomized)
    """
    def __init__(self, profile=None):
        self.profile = profile
        self._set_profile(profile)

    def _set_profile(self, profile):
        p = profile
        if p is None:
            p = random.choice(['novice','intermediate','expert'])
        self.profile = p
        if p == 'novice':
            self.skill = 0.2
            self.speed = 5.0
        elif p == 'intermediate':
            self.skill = 0.5
            self.speed = 3.0
        else:
            self.skill = 0.85
            self.speed = 1.5

    def reset(self):
        # keep same profile; could randomize
        pass

    def answer_question(self, difficulty):
        # difficulty: 0=easy,1=medium,2=hard
        # probability of correct depends on skill and difficulty
        diff_pen = {0: -0.3, 1: 0.0, 2: 0.4}[difficulty]
        # logistic
        prob = 1.0 / (1.0 + math.exp(- ( (self.skill - diff_pen) * 3.0 )))
        prob = max(0.01, min(0.99, prob))
        correct = random.random() < prob
        # time: base speed +/- noise, harder questions take longer
        time = max(0.2, random.gauss(self.speed + 0.5 * difficulty, 0.5))
        return correct, time
