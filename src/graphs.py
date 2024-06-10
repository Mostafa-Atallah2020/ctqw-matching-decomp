import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from sympy import symbols

from src import MCRX, GraphDrawer
from src.misc import binary_tuple_to_int_tuple


class StaticGraph:
    def __init__(self, edges) -> None:
        self._validate_edges(edges)
        self.edges = edges
        self.n_qubits = len(next(iter(edges))[0])
        self.nodes = set(range(2**self.n_qubits))
        self.rot_angle = self.__get_rot_angle()
        self.__int_edges = set([binary_tuple_to_int_tuple(t) for t in self.edges])

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def _validate_edges(self, edges):
        if not isinstance(edges, set):
            raise ValueError("Edges must be a set.")
        edge_length = None
        for edge in edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise ValueError("Each edge must be a tuple of length 2.")
            for node in edge:
                if not isinstance(node, str) or not all(bit in "01" for bit in node):
                    raise ValueError(
                        "Each node in an edge must be a string consisting of 0s and 1s."
                    )
                if edge_length is None:
                    edge_length = len(node)
                elif len(node) != edge_length:
                    raise ValueError("All nodes in edges must have the same length.")

    def __add__(self, other):
        return StaticGraph(self.nodes | other.nodes, self.edges | other.edges)

    def __get_rot_angle(self):
        # TODO: this one should not be fixed it should depend on the amplitudes of the edges
        # we will assume it is constant for simplicity.
        return np.pi / 2

    def get_adj_mat(self):
        return nx.adjacency_matrix(self.graph).todense()

    def get_statevector(self):
        vec = [0 for i in range(len(self.nodes))]
        for edge in self.__int_edges:
            a, b = edge
            if vec[a] == 0:
                vec[a] = symbols(f"alpha{a}")
            if vec[b] == 0:
                vec[b] = symbols(f"alpha{b}")

        return np.array(vec)

    def draw(self):
        GraphDrawer(self.n_qubits, self.__int_edges).show()


class ParallelEdgeGraph(StaticGraph):
    def __init__(self, edges):
        self._validate_parallel_edges(edges)
        super().__init__(edges)
        self.target = None
        self.vars = self.__get_vars()
        self.expr = self.__get_expr()

    def _validate_parallel_edges(self, edges):
        for edge in edges:
            i, j = edge
            diff_count = sum(1 for x, y in zip(i, j) if x != y)
            if diff_count != 1:
                raise ValueError("The Graph is not a Parallel Edge Graph.")

    def get_qc(self, simplified=False):
        mcrx = MCRX(self.n_qubits, self.expr, self.target, self.rot_angle)

        if simplified:
            return mcrx.simplify().qc
        else:
            return mcrx.qc

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


class NonDiagonalEdgeGraph(StaticGraph):
    def __init__(self, edges, simplified=False):
        super().__init__(edges)
        self.edge_sets = self.__split_tuples_by_changing_bit()
        self.__qc_dict = self.__get_qc_dict(simplified=simplified)

    def get_qc(self):
        circ = QuantumCircuit(self.n_qubits)
        keys = sorted(self.__qc_dict.keys())
        for idx in keys:
            qc = self.__qc_dict[idx]
            circ = circ.compose(qc, range(self.n_qubits))

        return circ

    def __split_tuples_by_changing_bit(self):
        # Initialize a dictionary to store subsets based on the changing bit position
        subsets = {}

        # Iterate through each tuple in the set
        for t in self.edges:
            # Find the position where the bits differ
            for i in range(len(t[0])):
                if t[0][i] != t[1][i]:
                    changing_bit_position = i
                    break

            # Add the tuple to the corresponding subset
            if changing_bit_position not in subsets:
                subsets[changing_bit_position] = set()

            subsets[changing_bit_position].add(t)

        return subsets

    def __get_qc_dict(self, simplified=False):
        qc_dict = {}

        for idx, edges in self.edge_sets.items():
            G = ParallelEdgeGraph(edges)
            qc = G.get_qc(simplified=simplified)
            qc_dict[idx] = qc

        return qc_dict
