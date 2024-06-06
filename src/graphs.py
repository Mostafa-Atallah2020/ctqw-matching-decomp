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
        # TODO: Find alternative implementation
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
        # positions = self._get_fixed_positions()
        labels = {node: format(node, f"0{self.n_qubits}b") for node in self.nodes}

        nx.draw(
            self.graph,
            labels=labels,
            with_labels=True,
            node_size=100 * self.n_qubits,
            node_color="skyblue",
            font_size=5,
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


class ParallelEdgeGraph(StaticGraph):
    def __init__(self, n_qubits, edges):
        super().__init__(n_qubits, edges)
        self.target = None
        self.edges = edges
        self.vars = self.__get_vars()
        self.expr = self.__get_expr()

    def __get_vars(self):
        sym_vars = []
        for i in range(self.n_qubits):
            sym_name = f"x_{i}"
            sym_vars.append(symbols(sym_name, latex=True))

        return sym_vars

    def __get_expr(self):
        expr = False
        for i, j in self.edges:
            subexpr = True
            for k in range(self.n_qubits):
                if (i[k] == j[k]) and (j[k] == "0"):
                    subexpr = subexpr & ~self.vars[k]
                elif (i[k] == j[k]) and (j[k] == "1"):
                    subexpr = subexpr & self.vars[k]
                else:
                    self.target = k
            expr = expr | subexpr

        return expr
