import numpy as np
import pandas as pd
from tqdm import trange
from QLearning import QLearning

def load_data():
    data = pd.read_csv("data/small.csv")
    S = list(data.s - 1)
    A = list(data.a - 1)
    R = list(data.r)
    SPR = list(data.sp - 1)
    return S, A, R, SPR, data.shape[0], data.shape[1] - 1

def main():
    k = 100
    gamma = 0.95
    state = 100
    action = 4
    S, A, R, SPR, n_state, n_action = load_data()

    alpha = 1 / k
    q_learning = QLearning(S, A, gamma, alpha, state, action)

    for _ in trange(k):
        for i in range(len(S)):
            q_learning.update(S[i], A[i], R[i], SPR[i])

    policy = q_learning.get_policy()
    with open("small.policy", 'w') as f:
        for p in policy:
            f.write(f"{p}\n")


if __name__ == "__main__":
    main()