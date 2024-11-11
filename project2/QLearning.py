import numpy as np


class QLearning:
    def __init__(self, S, A, gamma, alpha, state, action):
        self.S = S
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((state, action))

    def lookahead(self, s, a):
        return self.Q[s, a]

    def update(self, s, a, r, s_prime):
        max_future_q = np.max(self.Q[s_prime])
        self.Q[s, a] += self.alpha * (r + (self.gamma * max_future_q) - self.Q[s, a])

    def get_policy(self):
        return np.argmax(self.Q, axis=1) + 1