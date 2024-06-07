import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sympy import symbols

from src.misc import binary_tuple_to_int_tuple
from src import GraphDrawer


class StaticGraph:
    def __init__(self, n_qubits, edges) -> None:
        self.n_qubits = n_qubits
        self.nodes = set(range(2**n_qubits))
        self.edges = edges

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def __add__(self, other):
        return StaticGraph(self.nodes | other.nodes, self.edges | other.edges)

    def get_adj_mat(self):
        return nx.adjacency_matrix(self.graph).todense()

    def get_statevector(self):
        edges = set([binary_tuple_to_int_tuple(t) for t in self.edges])
        vec = [0 for i in range(len(self.nodes))]
        for edge in edges:
            a, b = edge
            if vec[a] == 0:
                vec[a] = symbols(f"alpha{a}")
            if vec[b] == 0:
                vec[b] = symbols(f"alpha{b}")

        return np.array(vec)

    def draw(self):
        edges = set([binary_tuple_to_int_tuple(t) for t in self.edges])
        GraphDrawer(self.n_qubits, edges).show()


class ParallelEdgeGraph(StaticGraph):
    def __init__(self, n_qubits, edges):
        super().__init__(n_qubits, edges)
        self.target = None
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
