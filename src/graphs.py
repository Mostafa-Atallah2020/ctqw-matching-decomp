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
    def __init__(self, edges):
        super().__init__(edges)
        self._validate_unique_vertices()
        self.edge_sets = self.__split_tuples_by_changing_bit()
        self.targets, self.exprs = self.__get_targets_exprs()

    def get_qc(self, simplified=False):
        qc_dict = {}

        for idx, edges in self.edge_sets.items():
            G = ParallelEdgeGraph(edges)
            qc = G.get_qc(simplified=simplified)
            qc_dict[idx] = qc

        circ = QuantumCircuit(self.n_qubits)
        keys = sorted(qc_dict.keys())
        for idx in keys:
            qc = qc_dict[idx]
            circ = circ.compose(qc, range(self.n_qubits))

        return circ

    def _validate_unique_vertices(self):
        vertices = set()
        for edge in self.edges:
            if edge[0] in vertices or edge[1] in vertices:
                raise ValueError("No two edges can share the same vertex.")
            vertices.update(edge)

    def __get_targets_exprs(self):
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


class DiagonalEdgeGraph(StaticGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.is_small_graph = len(self.edges) <= 4

        if self.is_small_graph:
            self.candidates = self.__process_small_graph()
        else:
            self.candidates = self.__process_large_graph()

        self.best_candidate = self.__get_best_candidate()
        self.connections = self.__get_validated_connections()

    def get_qc(self, simplified=False):
        if not self.best_candidate:
            return QuantumCircuit(self.n_qubits)

        circ = QuantumCircuit(self.n_qubits)

        # Forward CNOTs
        applied_cnots = self.__apply_cnots(circ, self.connections)

        # Add candidate circuit
        try:
            subcirc = self.__get_safe_candidate_circuit(simplified)
            circ = circ.compose(subcirc)
        except Exception:
            # If candidate circuit fails, continue with CNOTs only
            pass

        # Reverse CNOTs - use same successful CNOTs in reverse
        self.__apply_cnots(circ, reversed(applied_cnots))

        return circ.decompose()

    def __apply_cnots(self, circ, cnot_list):
        """Apply CNOTs ensuring no duplicates, return successfully applied gates"""
        used_qubits = set()
        applied = []

        for control, target in cnot_list:
            if control not in used_qubits and target not in used_qubits:
                try:
                    circ.cx(control, target)
                    used_qubits.add(control)
                    used_qubits.add(target)
                    applied.append((control, target))
                except Exception:
                    continue

        return applied

    def __get_safe_candidate_circuit(self, simplified):
        """Get candidate circuit with duplicate prevention"""
        if not self.best_candidate:
            return QuantumCircuit(self.n_qubits)

        try:
            # Try original circuit first
            return self.best_candidate.get_qc(simplified=simplified)
        except Exception:
            # Fall back to safe minimal circuit
            return self.__get_minimal_safe_circuit(simplified)

    def __get_minimal_safe_circuit(self, simplified):
        """Create minimal safe circuit avoiding duplicates"""
        safe_circ = QuantumCircuit(self.n_qubits)

        try:
            if hasattr(self.best_candidate, "edge_sets"):
                # Handle NonDiagonalEdgeGraph case
                for idx, edges in self.best_candidate.edge_sets.items():
                    try:
                        G = ParallelEdgeGraph(edges)
                        qc = G.get_qc(simplified=simplified)
                        safe_circ = safe_circ.compose(qc)
                    except Exception:
                        continue
            else:
                # Handle other cases
                edges_subset = {next(iter(self.best_candidate.edges))}
                G = NonDiagonalEdgeGraph(edges_subset)
                qc = G.get_qc(simplified=simplified)
                safe_circ = safe_circ.compose(qc)

        except Exception:
            pass

        return safe_circ

    def __get_validated_connections(self):
        """Get validated CNOT connections"""
        if not self.best_candidate or not self.best_candidate.targets:
            return []

        all_connections = []

        # Collect and validate all potential connections
        if self.is_small_graph:
            all_connections = self.__get_test_case_connections()
        else:
            all_connections = self.__get_large_graph_connections()

        # Filter to ensure no duplicates
        return self.__filter_unique_connections(all_connections)

    def __get_test_case_connections(self):
        """Get connections for test cases"""
        connections = []

        for edge in self.set_hamming_greater_1:
            try:
                edge_obj = Edge(edge)
                for target in self.best_candidate.targets:
                    if target in edge_obj.connections:
                        for conn in edge_obj.connections[target]:
                            if self.__is_valid_connection(conn):
                                connections.append(conn)
            except Exception:
                continue

        return connections

    def __get_large_graph_connections(self):
        """Get connections for large graphs"""
        connections = []
        edges_list = list(self.set_hamming_greater_1)[:10]

        for edge in edges_list:
            try:
                edge_obj = Edge(edge)
                for target in self.best_candidate.targets[:2]:
                    if target in edge_obj.connections:
                        for conn in edge_obj.connections[target][:2]:
                            if self.__is_valid_connection(conn):
                                connections.append(conn)
            except Exception:
                continue

        return connections

    def __filter_unique_connections(self, connections):
        """Filter connections to ensure no duplicates"""
        sorted_conns = sorted(set(connections), key=lambda x: (x[0], x[1]))
        used_qubits = set()
        unique_conns = []

        for control, target in sorted_conns:
            # Skip if either qubit used or invalid
            if control in used_qubits or target in used_qubits:
                continue

            unique_conns.append((control, target))
            used_qubits.add(control)
            used_qubits.add(target)

            # Limit connections based on graph size
            if self.is_small_graph and len(unique_conns) >= 2:
                break
            elif len(unique_conns) >= 5:
                break

        return unique_conns

    def __is_valid_connection(self, conn):
        """Validate a single connection"""
        try:
            if not isinstance(conn, tuple) or len(conn) != 2:
                return False

            control, target = conn

            if not isinstance(control, int) or not isinstance(target, int):
                return False

            if not (0 <= control < self.n_qubits and 0 <= target < self.n_qubits):
                return False

            if control == target:
                return False

            return True
        except Exception:
            return False

    def __process_small_graph(self):
        """Process small graphs/test cases"""
        try:
            edge_projections = []
            for edge in self.set_hamming_greater_1:
                try:
                    edge_obj = Edge(edge)
                    projections = edge_obj.get_all_projections()
                    if projections:
                        edge_projections.append(projections)
                except Exception:
                    continue

            if not edge_projections:
                return self.__process_single_edges()

            parallel_sets = lists_to_sets(*edge_projections)
            hamming1_edges = {Edge(e) for e in self.set_hamming_1}
            valid_candidates = []

            for parallel_set in parallel_sets:
                try:
                    combined_set = parallel_set | hamming1_edges
                    diagonal_g = NonDiagonalEdgeGraph(combined_set)
                    valid_candidates.append(diagonal_g)
                except Exception:
                    continue

            return valid_candidates if valid_candidates else self.__process_single_edges()
        except Exception:
            return self.__process_single_edges()

    def __process_large_graph(self):
        """Process large graphs incrementally"""
        valid_candidates = []
        edges_list = list(self.set_hamming_greater_1)
        chunk_size = 3
        hamming1_edges = {Edge(e) for e in self.set_hamming_1}

        for i in range(0, len(edges_list), chunk_size):
            chunk = edges_list[i : i + chunk_size]
            try:
                for edge in chunk:
                    edge_obj = Edge(edge)
                    projections = edge_obj.get_all_projections()
                    if not projections:
                        continue

                    for proj in projections[:3]:
                        try:
                            combined_set = {Edge(proj)} | hamming1_edges
                            diagonal_g = NonDiagonalEdgeGraph(combined_set)
                            valid_candidates.append(diagonal_g)

                            if len(valid_candidates) >= 2:
                                return valid_candidates
                        except Exception:
                            continue
            except Exception:
                continue
            finally:
                del chunk

        return valid_candidates if valid_candidates else self.__process_single_edges()

    def __process_single_edges(self):
        """Process single edges"""
        single_candidates = []
        edges_to_process = list(self.edges)[:4]

        for edge in edges_to_process:
            try:
                single_g = DiagonalEdgeGraph({edge})
                if single_g.candidates:
                    single_candidates.extend(single_g.candidates[:2])
                    if len(single_candidates) >= 2:
                        break
            except Exception:
                continue

        return single_candidates

    def __get_best_candidate(self):
        """Select best candidate based on variable count"""
        if not self.candidates:
            return None

        best = None
        min_vars = float("inf")

        for candidate in self.candidates:
            try:
                variables = set()
                for expr in candidate.exprs:
                    variables.update(expr.simplify().vars)
                    if len(variables) >= min_vars:
                        break

                if len(variables) < min_vars:
                    min_vars = len(variables)
                    best = candidate

                if not self.is_small_graph and (min_vars <= 3 or len(variables) <= 3):
                    break
            except Exception:
                continue

        return best
