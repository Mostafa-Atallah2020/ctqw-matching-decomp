import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from sympy import symbols

from src import MCRX, Edge, GraphDrawer
from src.misc import binary_tuple_to_int_tuple, hamming_distance, lists_to_sets


class StaticGraph:
    def __init__(self, edges) -> None:
        self.edges = set()
        self._validate_edges(edges)
        self._validate_unique_vertices()

        self.n_qubits = len(next(iter(self.edges))[0])
        self.nodes = set(range(2**self.n_qubits))
        self.rot_angle = self.__get_rot_angle()
        self.__int_edges = set([binary_tuple_to_int_tuple(t) for t in self.edges])

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def _validate_edges(self, edges):
        if not isinstance(edges, set):
            raise ValueError("Edges must be a set.")

        for e in edges:
            if isinstance(e, Edge):
                self.edges.add(e.edge)
            elif isinstance(e, tuple):
                self.edges.add(e)
            else:
                raise ValueError("Edge type should be a tuple or Edge")

    def _validate_unique_vertices(self):
        vertices = set()
        for edge in self.edges:
            if edge[0] in vertices or edge[1] in vertices:
                raise ValueError("No two edges can share the same vertex.")
            vertices.update(edge)

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
            if hamming_distance(i, j) != 1:
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


class DiagonalEdgeGraph(StaticGraph):
    def __init__(self, edges, simplified=False):
        super().__init__(edges)
        self.set_hamming_1, self.set_hamming_greater_1 = self.__split_by_hamming_distance()
        self.candidates = self.__get_candidates()

    def __split_by_hamming_distance(self):
        """Split edges into sets based on Hamming distance."""
        set_hamming_1 = set()
        set_hamming_greater_1 = set()

        for edge in self.edges:
            dist = hamming_distance(edge[0], edge[1])
            if dist == 1:
                set_hamming_1.add(edge)
            else:
                set_hamming_greater_1.add(edge)

        return set_hamming_1, set_hamming_greater_1

    def __get_candidates(self):
        parallel_candidates = []
        for e in self.set_hamming_greater_1:
            edge = Edge(e)
            parallel_candidates.append(edge.get_parallel_candidates())

        parallel_candidates = lists_to_sets(*parallel_candidates)

        non_diagonal_candidates = []
        for c in parallel_candidates:
            non_diagonal_candidates.append(c | {Edge(e) for e in self.set_hamming_1})

        valid_candidates = []
        for g in non_diagonal_candidates:
            try:
                diagonalG = NonDiagonalEdgeGraph(g)
                valid_candidates.append(diagonalG)

            except:
                # Skip the item that caused an error
                continue

        return valid_candidates
