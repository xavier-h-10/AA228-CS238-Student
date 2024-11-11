import numpy as np
import random
from collections import deque

class DoubleQLearning:
    def __init__(self, S, A, gamma, alpha, state, action):
        self.S = S
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.buffer = deque(maxlen=10000)
        self.n_step = 3
        self.Q1 = np.zeros((state, action))
        self.Q2 = np.zeros((state, action))

    def store_transition(self, transition):
        self.buffer.append(transition)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def lookahead(self, s, a):
        return self.Q1[s, a] + self.Q2[s, a]

    def update(self, s, a, r, s_prime):
        if random.random() < 0.5:
            max_a_prime = np.argmax(self.Q1[s_prime])
            self.Q1[s, a] += self.alpha * (r + self.gamma * self.Q2[s_prime, max_a_prime] - self.Q1[s, a])
        else:
            max_a_prime = np.argmax(self.Q2[s_prime])
            self.Q2[s, a] += self.alpha * (r + self.gamma * self.Q1[s_prime, max_a_prime] - self.Q2[s, a])

    def multi_step_update(self, transitions):
        G = sum([transitions[i][2] * (self.gamma ** i) for i in range(self.n_step)])
        s, a = transitions[0][0], transitions[0][1]
        s_prime = transitions[-1][3]
        self.update(s, a, G, s_prime)

    def get_policy(self):
        return np.argmax(self.Q1 + self.Q2, axis=1) + 1