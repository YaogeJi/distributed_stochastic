import numpy as np
from networkx import *
from utils import MaxIterError
import torch


class FullyConnectedNetwork:
    def __init__(self, m):
        self.m = m
        self.w = 1/m * np.ones((m, m))

    def generate(self):
        return self.w

class ErodoRenyi:
    def __init__(self, m, rho, seed):
        self.node = m
        self.rho = rho
        assert self.rho <= 1 and self.rho >= 0
        self.probability = 1 - self.rho
        self.seed = seed

    def generate(self):
        connected = False
        connectivity = 1
        print("network generating")
        seed = self.seed
        for i in range(10000):
            G = erdos_renyi_graph(self.node, self.probability, seed=seed)
            seed += 1
            connected = is_connected(G)
            if not connected:
                continue
            adjacent_matrix = to_numpy_array(G)
            matrix = np.zeros((self.node, self.node))
            for i in G.edges:
                degree = max(G.degree(i[0]), G.degree(i[1]))
                G.add_edge(*i, weight=1/(degree + 1))
            adjacent_matrix = to_numpy_array(G)
            weighted_matrix = np.eye(self.node) - np.diag(sum(adjacent_matrix)) + adjacent_matrix
            if self.node == 1:
                return torch.tensor(weighted_matrix, dtype=torch.float)
            else:
                eigenvalue, _ = np.linalg.eig(weighted_matrix)
                sorted_eigenvalue = np.sort(np.abs(eigenvalue))
                connectivity = sorted_eigenvalue[-2]
                print(connectivity, self.rho, self.probability)
                if connectivity - self.rho > 0.001:
                    self.probability += 0.01
                elif connectivity - self.rho < -0.001:
                    self.probability -= 0.01
                elif np.abs(connectivity - self.rho) < 0.001:
                    print("generating network succeed")
                    return torch.tensor(weighted_matrix, dtype=torch.float)
        else:
            raise MaxIterError("achieve max iteration without achieving target connectivity")
