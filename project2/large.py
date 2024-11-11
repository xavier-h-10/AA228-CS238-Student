import numpy as np
import pandas as pd
from tqdm import trange
import bisect

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


def load_data():
    data = pd.read_csv("data/large.csv")
    S = list(data.s - 1)
    A = list(data.a - 1)
    R = list(data.r)
    SPR = list(data.sp - 1)
    return S, A, R, SPR, data.shape[0], data.shape[1] - 1


def find_missing_states(n_state, set_S):
    missing_states = {i for i in range(n_state) if i not in set_S}
    print(f"Number of missing states: {len(missing_states)}")
    return missing_states


def find_nearest_neighbors(missing_states, unique_S):
    nearest_neighbor = {}
    for i in missing_states:
        index_right = bisect.bisect_right(unique_S, i)
        index_left = index_right - 1

        if index_right == len(unique_S):
            min_index = index_left
        elif index_left == -1:
            min_index = index_right
        else:
            if unique_S[index_right] - i > i - unique_S[index_left]:
                min_index = index_left
            else:
                min_index = index_right

        nearest_neighbor[i] = unique_S[min_index]
    return nearest_neighbor


def main():
    k = 100
    gamma = 0.95
    S, A, R, SPR, n_state, n_action = load_data()
    set_S = set(S)
    state = 302020
    action = 9

    alpha = 1 / k
    q_learning = QLearning(S, A, gamma, alpha, state, action)

    missing_states = find_missing_states(n_state, set_S)
    unique_S = sorted(set_S)
    nearest_neighbors = find_nearest_neighbors(missing_states, unique_S)

    for _ in trange(k):
        for i in range(len(S)):
            q_learning.update(S[i], A[i], R[i], SPR[i])

    policy = q_learning.get_policy()
    with open("large.policy", 'w') as f:
        for i, p in enumerate(policy):
            if i in missing_states:
                f.write(f"{policy[nearest_neighbors[i]]}\n")
            else:
                f.write(f"{p}\n")


if __name__ == "__main__":
    main()
