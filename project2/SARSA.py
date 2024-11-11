import numpy as np
import random

class SARSA:
    def __init__(self, n_states, n_actions, gamma, alpha, epsilon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.Q[state])

    def update(self, s, a, r, s_prime, a_prime):
        td_target = r + self.gamma * self.Q[s_prime, a_prime]
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def get_policy(self):
        return np.argmax(self.Q, axis=1) + 1