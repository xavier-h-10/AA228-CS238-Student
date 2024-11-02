import sys
import pandas as pd
import networkx as nx
import numpy as np
from abc import ABC, abstractmethod

from networkx.generators.classic import null_graph
from scipy.special import loggamma
from networkx.drawing.nx_agraph import write_dot
import os
import time
import random
from tqdm import tqdm


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


class Variable():
    def __init__(self, name: str, r: int):
        self.name = name
        self.r = r

    def __str__(self):
        return "(" + self.name + ", " + str(self.r) + ")"


def statistics(vars: list[Variable], graph: nx.DiGraph, data: np.ndarray) -> list[np.ndarray]:
    n = len(vars)
    r = np.array([var.r for var in vars])
    q = np.array([int(np.prod([r[j] for j in graph.predecessors(i)])) for i in range(n)])
    M = [np.zeros((q[i], r[i])) for i in range(n)]

    for o in data.T:
        for i in range(n):
            k = o[i]
            parents = list(graph.predecessors(i))
            j = 0
            if len(parents) != 0:
                j = np.ravel_multi_index(o[parents], r[parents])
            M[i][j, k] += 1.0
    return M


def prior(variables: list[Variable], graph: nx.DiGraph) -> list[np.ndarray]:
    n = len(variables)
    r = [var.r for var in variables]
    q = np.array([int(np.prod([r[j] for j in graph.predecessors(i)])) for i in range(n)])
    return [np.ones((q[i], r[i])) for i in range(n)]


def bayesian_score_component(M: np.ndarray, alpha: np.ndarray) -> float:
    p = np.sum(loggamma(alpha + M))
    p -= np.sum(loggamma(alpha))
    p += np.sum(loggamma(np.sum(alpha, axis=1)))
    p -= np.sum(loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p


def bayesian_score(vars: list[Variable], G: nx.DiGraph, D: np.ndarray) -> float:
    n = len(vars)
    M = statistics(vars, G, D)
    alpha = prior(vars, G)
    return np.sum([bayesian_score_component(M[i], alpha[i]) for i in range(n)])


class DirectedGraphSearchMethod(ABC):
    @abstractmethod
    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        pass


class K2Search():
    def __init__(self, ordering: list[int]):
        self.ordering = ordering

    def fit(self, variables: list[Variable], data: np.ndarray) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(variables)))
        for k, i in tqdm(enumerate(self.ordering[1:])):
            y = bayesian_score(variables, graph, data)
            while True:
                y_best, j_best = -np.inf, 0
                for j in self.ordering[:k]:
                    if not graph.has_edge(j, i):
                        graph.add_edge(j, i)
                        y_prime = bayesian_score(variables, graph, data)
                        if y_prime > y_best:
                            y_best, j_best = y_prime, j
                        graph.remove_edge(j, i)
                if y_best > y:
                    y = y_best
                    graph.add_edge(j_best, i)
                else:
                    break
        return graph


def compute(infile, outfile):
    df = pd.read_csv(infile)
    data = df.to_numpy().T - 1
    columns = df.columns

    value_counts = {column: max(df[column]) for column in df.columns}
    variables = [Variable(key, value) for key, value in value_counts.items()]

    num_columns = df.shape[1]

    best_graph = None
    best_score = 0

    for i in range(10):
        ordering = list(range(num_columns))
        random.shuffle(ordering)
        M = K2Search(ordering)

        start_time = time.time()
        graph = M.fit(variables, data)
        end_time = time.time()

        score = bayesian_score(variables, graph, data)

        print("score:", score)
        print(f"Elapsed time: {end_time - start_time} seconds")

        if score > best_score or best_graph == None:
            best_graph = graph
            best_score = score



    labels = {i: columns[i] for i in range(len(columns))}

    graph = nx.relabel_nodes(best_graph, labels)

    with open(outfile, 'w') as f:
        for u, v in graph.edges():
            f.write(f"{u},{v}\n")

    dot_file = 'graph.dot'
    write_dot(graph, dot_file)

    os.system(f'dot -Tpng {dot_file} -o graph.png')


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
