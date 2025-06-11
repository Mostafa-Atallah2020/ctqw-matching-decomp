import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from sympy import symbols

from src import MCRX, Edge, Expression, GraphDrawer
from src.mcrx_simplifier import MCRXCascadeSimplifier
from src.misc import (
    binary_tuple_to_int_tuple,
    get_cyclic_connections,
    graph_matchings,
    hamming_distance,
    lists_to_sets,
)


class Graph:
    def __init__(self, edges) -> None:
        self.edges = set()
        self._validate_edges(edges)

        if self.edges:
            first_edge = next(iter(self.edges))
            if isinstance(first_edge[0], str):
                self.n_qubits = len(first_edge[0])
            else:
                self.n_qubits = len(bin(max(max(self.edges)))) - 2
        else:
            raise ValueError("Empty edge set")

        self.nodes = set(range(2**self.n_qubits))
        self.rot_angle = np.pi / 2
        self.__int_edges = set([self._edge_to_int_tuple(e) for e in self.edges])
        self.set_hamming_1, self.set_hamming_greater_1 = self.__split_by_hamming_distance()

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.__int_edges)

    def _edge_to_int_tuple(self, edge):
        if isinstance(edge[0], str):
            return tuple(int(v, 2) for v in edge)
        return edge

    def _validate_edges(self, edges):
        if isinstance(edges, set):
            self.edges = edges
        elif isinstance(edges, list):
            self.edges = set(edges)
        else:
            raise ValueError("Edges must be a set or list.")

        for e in self.edges:
            if not (isinstance(e, tuple) and len(e) == 2):
                raise ValueError("Each edge should be a tuple of length 2")

    def __split_by_hamming_distance(self):
        set_hamming_1 = set()
        set_hamming_greater_1 = set()

        for edge in self.__int_edges:
            dist = bin(edge[0] ^ edge[1]).count("1")
            if dist == 1:
                set_hamming_1.add(edge)
            else:
                set_hamming_greater_1.add(edge)

        return set_hamming_1, set_hamming_greater_1

    def __repr__(self):
        return f"StaticGraph(edges={self.edges})"


class PowerOf2EdgeGraph(Graph):
    def __init__(self, edges):
        super().__init__(edges)
        self.subgraphs = self.__decompose_into_subgraphs()

    def __decompose_into_subgraphs(self):
        edges = list(self.graph.edges)
        subgraphs = []

        while edges:
            k = int(np.log2(len(edges)))
            subgraph_size = 2**k

            # Select a random edge as a starting point
            start_edge = random.choice(edges)
            subgraph_edges = [start_edge]
            edges.remove(start_edge)

            # Greedily add edges that don't share vertices with existing edges
            for _ in range(subgraph_size - 1):
                if not edges:
                    break
                for edge in edges:
                    if all(len(set(edge) & set(e)) == 0 for e in subgraph_edges):
                        subgraph_edges.append(edge)
                        edges.remove(edge)
                        break

            subgraphs.append(nx.Graph(subgraph_edges))

        return subgraphs


class StaticGraph:
    def __init__(self, edges) -> None:
        self.edges = set()
        self._validate_edges(edges)

        self.n_qubits = len(next(iter(self.edges))[0])
        self.nodes = set(range(2**self.n_qubits))
        self.rot_angle = np.pi / 2
        self.__int_edges = set([binary_tuple_to_int_tuple(t) for t in self.edges])
        self.set_hamming_1, self.set_hamming_greater_1 = self.__split_by_hamming_distance()

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def __repr__(self):
        return "StaticGraph(%s)" % (self.edges)

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

    def __add__(self, other):
        return StaticGraph(self.nodes | other.nodes, self.edges | other.edges)

    def get_adj_mat(self):
        vertex_to_index = {v: i for i, v in enumerate(self.nodes)}
        num_vertices = len(self.nodes)
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

        for edge in self.__int_edges:
            v1, v2 = edge
            if v1 in vertex_to_index and v2 in vertex_to_index:
                i, j = vertex_to_index[v1], vertex_to_index[v2]
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1  # For undirected graph

        return adj_matrix

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


class DynamicGraph:
    def __init__(self, graph_sequence) -> None:
        """
        Initialize the DynamicGraph with a sequence of (graph, time) tuples.
        Ensures all graphs have the same number of qubits.

        Parameters:
        graph_sequence (list of tuples): Each tuple contains a graph object and a corresponding time.
        """
        self.graph_sequence = graph_sequence
        self.n_qubits = self.graph_sequence[0][0].n_qubits

        # Validate that all graphs have the same number of qubits
        for graph, _ in self.graph_sequence:
            if graph.n_qubits != self.n_qubits:
                raise ValueError("All graphs in the sequence must have the same number of qubits")

    def time_evo_op(self, t_steps=1):
        """
        Compute the time evolution operator for the sequence of graphs.

        Returns:
        Operator: The resulting time evolution operator.
        """
        time_evo_op = np.eye(2**self.n_qubits, dtype=complex)

        for _ in range(t_steps):
            for graph, time in self.graph_sequence:
                adj_matrix = graph.get_adj_mat()
                unitary = expm(-1j * adj_matrix * time)
                time_evo_op = np.dot(unitary, time_evo_op)

        time_evo_op = Operator(time_evo_op)
        return time_evo_op

    def draw(self):
        for i, (graph, delta_t) in enumerate(self.graph_sequence):
            print(f"{graph} | Time = {delta_t}")
            graph.draw()
            plt.show()


class IntersectingEdgesGraph(StaticGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.subgraphs = self.__decompose_into_subgraphs()

    def __decompose_into_subgraphs(self):
        decomposed_subgraphs = []

        subgraphs = graph_matchings(self.edges)
        for sg in subgraphs:
            g = MultiEdgeGraph(sg)
            decomposed_subgraphs.append(g)
        return decomposed_subgraphs


class MultiEdgeGraph(StaticGraph):
    def __new__(cls, edges):
        temp_instance = super().__new__(cls)
        StaticGraph.__init__(temp_instance, edges)

        if len(temp_instance.set_hamming_greater_1) == 0:
            return NonDiagonalEdgeGraph(edges)
        else:
            return DiagonalEdgeGraph(edges)


class ParallelEdgeGraph(StaticGraph):
    def __init__(self, edges):
        self._validate_parallel_edges(edges)
        super().__init__(edges)
        self.target = None
        self.vars = self.__get_vars()
        self.expr = Expression(self.__get_expr())

    def _validate_parallel_edges(self, edges):
        for edge in edges:
            i, j = edge
            if hamming_distance(i, j) != 1:
                raise ValueError("The Graph is not a Parallel Edge Graph.")

    def get_qc(self, simplified=False):
        if self.expr.expr.simplify() == True:
            qc = QuantumCircuit(self.n_qubits)
            qc.rx(self.rot_angle, self.target)
            return qc
        else:
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
    """
    A graph class for handling non-diagonal edges in quantum circuits.
    Non-diagonal edges represent transitions between quantum states that differ by
    a Hamming distance of 1.
    """

    def __init__(self, edges):
        super().__init__(edges)
        self._validate_edge_distances()
        self._validate_vertex_usage()
        self.edge_sets = self.__split_tuples_by_changing_bit()
        self.targets, self.exprs = self.__get_targets_exprs()

    def _validate_edge_distances(self):
        """Ensure all edges have Hamming distance of 1."""
        for edge in self.edges:
            dist = sum(b1 != b2 for b1, b2 in zip(edge[0], edge[1]))
            if dist != 1:
                raise ValueError(f"Edge {edge} has Hamming distance {dist}, expected 1")

    def _validate_vertex_usage(self):
        """Ensure each vertex appears in at most 2 edges (for implementable quantum circuits)."""
        vertex_count = defaultdict(int)
        for edge in self.edges:
            vertex_count[edge[0]] += 1
            vertex_count[edge[1]] += 1

        for vertex, count in vertex_count.items():
            if count > 2:
                raise ValueError(f"Vertex {vertex} appears in {count} edges, maximum allowed is 2")

    def get_qc(self, simplified=False):
        """
        Generate a quantum circuit implementing the graph transformations.

        Args:
            simplified (bool): Whether to simplify the resulting circuit

        Returns:
            QuantumCircuit: The constructed quantum circuit
        """
        qc_dict = {}

        # Build subcircuits for each edge set
        for idx, edges in self.edge_sets.items():
            G = ParallelEdgeGraph(edges)
            qc = G.get_qc(simplified=simplified)
            qc_dict[idx] = qc

        # Combine subcircuits in order
        circ = QuantumCircuit(self.n_qubits)
        for idx in sorted(qc_dict.keys()):
            qc = qc_dict[idx]
            circ = circ.compose(qc, range(self.n_qubits))

        return circ

    def __split_tuples_by_changing_bit(self):
        """
        Group edges by which qubit position changes.
        Returns:
            dict: Maps bit position to set of edges that change that bit
        """
        subsets = {}

        for t in self.edges:
            # Find position where bits differ
            for i, (b1, b2) in enumerate(zip(t[0], t[1])):
                if b1 != b2:
                    if i not in subsets:
                        subsets[i] = set()
                    subsets[i].add(t)
                    break  # Only one bit changes per edge

        return subsets

    def __get_targets_exprs(self):
        """
        Extract target qubits and expressions for MCRX gates.

        Returns:
            tuple: (list of target qubits, list of control expressions)
        """
        targets = []
        exprs = []

        for idx, edges in self.edge_sets.items():
            G = ParallelEdgeGraph(edges)
            target = G.target
            expr = G.expr

            if target not in targets:
                targets.append(target)
            if expr not in exprs:
                exprs.append(expr)

        return targets, exprs


class DiagonalEdgeGraph(StaticGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.candidates = self._get_candidates()
        self.best_candidate = self._get_best_candidate()
        self.connections = self._get_filtered_connections()

    def _get_filtered_connections(self):
        """Get connections for diagonal edges"""
        raw_connections = []

        # Get connections from each diagonal edge
        for edge in self.set_hamming_greater_1:
            edge_obj = Edge(edge)
            # Get differing positions
            diff_positions = []
            for i, (b1, b2) in enumerate(zip(edge_obj.start, edge_obj.end)):
                if b1 != b2:
                    diff_positions.append(i)

            if len(diff_positions) > 1:
                # Create connections from first differing position to all others
                first_pos = diff_positions[0]
                for pos in diff_positions[1:]:
                    raw_connections.append((first_pos, pos))

        return sorted(list(set(raw_connections)))

    def _get_candidates(self):
        """
        Get valid candidates for transformation with detailed debugging.
        """
        # print("Starting _get_candidates")
        if not self.set_hamming_greater_1:
            print("No diagonal edges found")
            return []

        valid_candidates = []
        hamming1_edges = self.set_hamming_1.copy()
        # print(f"Hamming-1 edges: {hamming1_edges}")
        # print(f"Diagonal edges: {self.set_hamming_greater_1}")

        # For each diagonal edge
        for diagonal_edge in self.set_hamming_greater_1:
            # print(f"\nProcessing diagonal edge: {diagonal_edge}")
            start, end = diagonal_edge

            # Find positions where bits differ
            diff_positions = []
            for i, (b1, b2) in enumerate(zip(start, end)):
                if b1 != b2:
                    diff_positions.append(i)
            # print(f"Differing positions: {diff_positions}")

            # For each differing position, create projections
            for target_qubit in diff_positions:
                # print(f"\nTrying target qubit {target_qubit}")
                projections = set()

                # Create projection for start node
                start_proj = list(start)
                start_proj[target_qubit] = end[target_qubit]
                start_intermediate = "".join(start_proj)
                projections.add((start, start_intermediate))

                # Create projection for end node
                end_proj = list(end)
                end_proj[target_qubit] = start[target_qubit]
                end_intermediate = "".join(end_proj)
                projections.add((end_intermediate, end))

                # print(f"Generated projections: {projections}")

                # Add projections to existing Hamming-1 edges
                try:
                    combined_edges = projections | hamming1_edges
                    # print(f"Combined edges: {combined_edges}")

                    # Let's see why NonDiagonalEdgeGraph might be failing
                    # print("Checking Hamming distances in combined edges:")
                    for edge in combined_edges:
                        dist = sum(1 for a, b in zip(edge[0], edge[1]) if a != b)
                        # print(f"Edge {edge}: Hamming distance = {dist}")

                    candidate_graph = NonDiagonalEdgeGraph(combined_edges)
                    valid_candidates.append(candidate_graph)
                    # print("Successfully created candidate")
                except ValueError as e:
                    print(f"Failed to create candidate: {str(e)}")
                    continue

        # print(f"\nFinal number of valid candidates: {len(valid_candidates)}")
        return valid_candidates

    def _get_best_candidate(self):
        """Select the best candidate based on the minimum number of variables."""
        if not self.candidates:
            return None

        min_vars = float("inf")
        best = None

        for candidate in self.candidates:
            try:
                variables = set()
                for expr in candidate.exprs:
                    variables.update(expr.simplify().vars)

                var_count = len(variables)
                if var_count < min_vars:
                    min_vars = var_count
                    best = candidate

            except Exception:
                continue

        return best

    def get_qc(self, simplified=False):
        """Generate quantum circuit for the graph."""
        if not self.best_candidate:
            return QuantumCircuit(self.n_qubits)

        # Get CNOT connections from cyclic path
        cnots_lists = []
        for target in self.best_candidate.targets:
            cnots = get_cyclic_connections(self.connections, target)
            for cx in cnots:
                if cx not in cnots_lists:
                    cnots_lists.append(cx)

        unsimplified_qc = self.best_candidate.get_qc()
        n_qubits = self.best_candidate.n_qubits

        # Build the circuit
        circ = QuantumCircuit(n_qubits)

        # Add forward CNOTs
        for t in cnots_lists:
            circ.cx(*t)

        if simplified:
            try:
                simplifier = MCRXCascadeSimplifier(verbose=False)
                simplified_qc, _ = simplifier.simplify(unsimplified_qc)
                circ.append(simplified_qc, range(n_qubits))
            except:
                circ.append(unsimplified_qc, range(n_qubits))
        else:
            circ.append(unsimplified_qc, range(n_qubits))

        # Add reverse CNOTs
        for t in reversed(cnots_lists):
            circ.cx(*t)

        return circ.decompose()

    def __del__(self):
        """Clean up any remaining resources"""
        attrs = ["candidates", "best_candidate", "connections"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
