import numpy as np
import pandas as pd
from tqdm import trange
import bisect

# Vanilla Q-Learning class with modified unexplored state handling
class QLearning:
    def __init__(self, n_states, n_actions, gamma, alpha):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((n_states, n_actions))

    def lookahead(self, s, a):
        return self.Q[s, a]

    def update(self, s, a, r, s_prime):
        max_future_q = np.max(self.Q[s_prime])
        self.Q[s, a] += self.alpha * (r + self.gamma * max_future_q - self.Q[s, a])

    def get_policy(self):
        return np.argmax(self.Q, axis=1) + 1  # +1 to adjust for 1-based action indexing

    def assign_action_based_on_velocity(self, pos, vel):
        # Assuming actions 0-3 accelerate left and actions 4-6 accelerate right
        if vel < 0:
            return 0  # Low acceleration (left) if velocity is negative
        else:
            return self.n_actions - 1  # High acceleration (right) if velocity is positive


def load_data(filename):
    data = pd.read_csv(f"data/{filename}.csv")
    S = list(data.s - 1)
    A = list(data.a - 1)
    R = list(data.r)
    SPR = list(data.sp - 1)
    return S, A, R, SPR, data.shape[0], data.shape[1] - 1


def find_missing_states(n_state, set_S):
    missing_states = {i for i in range(n_state) if i not in set_S}
    print(f"Number of missing states: {len(missing_states)}")
    return missing_states


def main(filename='medium', k=100, gamma=0.95, alpha=0.1):
    # Load data
    S, A, R, SPR, n_state, n_action = load_data(filename)
    set_S = set(S)

    n_state = 50000
    n_action = 7

    # Initialize Q-learning with parameters
    q_learning = QLearning(n_state, n_action, gamma, alpha)

    # Find missing states and their associated position and velocity
    missing_states = find_missing_states(n_state, set_S)

    # Handle unexplored states by assigning actions based on initial velocity association
    for state in missing_states:
        pos = state % 500       # Assuming 500 possible positions
        vel = state // 500       # Assuming 100 possible velocities
        initial_action = q_learning.assign_action_based_on_velocity(pos, vel)
        q_learning.Q[state, initial_action] = 0.1  # Small initialization to bias the action choice

    # Run Q-learning updates
    for _ in trange(k):
        for i in range(len(S)):
            q_learning.update(S[i], A[i], R[i], SPR[i])

    # Derive and save the policy
    policy = q_learning.get_policy()
    with open(f"{filename}.policy", 'w') as f:
        for i, p in enumerate(policy):
            f.write(f"{p}\n")


if __name__ == "__main__":
    main()
