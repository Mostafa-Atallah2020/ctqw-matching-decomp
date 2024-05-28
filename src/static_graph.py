import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sympy import symbols

from src.misc import binary_tuple_to_int_tuple


class StaticGraph:
    def __init__(self, n_qubits, edges) -> None:
        self.n_qubits = n_qubits
        self.nodes = set(range(2**n_qubits))
        self.edges = set([binary_tuple_to_int_tuple(t) for t in edges])

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def __add__(self, other):
        return StaticGraph(self.nodes | other.nodes, self.edges | other.edges)

    def get_adj_mat(self):
        return nx.adjacency_matrix(self.graph).todense()

    def get_statevector(self):
        vec = [0 for i in range(len(self.nodes))]
        for edge in self.edges:
            a, b = edge
            if vec[a] == 0:
                vec[a] = symbols(f"alpha{a}")
            if vec[b] == 0:
                vec[b] = symbols(f"alpha{b}")

        return np.array(vec)

    def draw(self):
        # TODO: It's better to make it a text drawer for now
        positions = self._get_fixed_positions()
        labels = {node: format(node, f"0{self.n_qubits}b") for node in self.nodes}

        nx.draw(
            self.graph,
            pos=positions,
            labels=labels,
            with_labels=True,
            node_size=250 * self.n_qubits,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            font_weight="bold",
            edge_color="gray",
            linewidths=10,
            alpha=0.7,
        )

        plt.show()

    def _get_fixed_positions(self):
        positions = {}
        for i, node in enumerate(sorted(self.nodes)):
            row = i // 2
            col = i % 2
            positions[node] = (col, row)

        return positions
