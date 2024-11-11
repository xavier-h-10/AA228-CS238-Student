import numpy as np
import pandas as pd
from tqdm import trange
import bisect
from QLearning import QLearning

def load_data():
    data = pd.read_csv(f"data/medium.csv")
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
    k = 1000
    gamma = 1.0
    S, A, R, SPR, n_state, n_action = load_data()
    set_S = set(S)
    state = 50000
    action = 7
    alpha = 0.05
    q_learning = QLearning(S, A, gamma, alpha, state, action)

    missing_states = find_missing_states(n_state, set_S)
    unique_S = sorted(set_S)
    nearest_neighbors = find_nearest_neighbors(missing_states, unique_S)

    for _ in trange(k):
        for i in range(len(S)):
            q_learning.update(S[i], A[i], R[i], SPR[i])

    policy = q_learning.get_policy()
    with open("medium.policy", 'w') as f:
        for i, p in enumerate(policy):
            if i in missing_states:
                f.write(f"{policy[nearest_neighbors[i]]}\n")
            else:
                f.write(f"{p}\n")


if __name__ == "__main__":
    main()
